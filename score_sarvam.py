#!/usr/bin/env python3
"""Compare paired TTS recordings using the same Sarvam batch-ASR model/settings."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import quote, urlsplit
import wave

import requests
from common import read_items, sha256, write_json
from score import aggregate, score_text

API = 'https://api.sarvam.ai/speech-to-text/job/v1'
TERMINAL = {'Completed', 'PartiallyCompleted', 'Failed'}


class SarvamBatch:
    def __init__(self, key, timeout=180):
        self.key, self.timeout = key, timeout
        self.session = requests.Session()

    def api(self, method, suffix='', body=None):
        try:
            for attempt in range(4):
                response = self.session.request(method, API + suffix, json=body,
                    headers={'api-subscription-key': self.key}, timeout=self.timeout)
                if response.status_code != 429 or attempt == 3:
                    break
                try:
                    if response.json().get('error', {}).get('code') == 'insufficient_quota_error':
                        break
                except (ValueError, AttributeError):
                    pass
                time.sleep(10 * (attempt + 1))
        except requests.RequestException as exc:
            raise RuntimeError(f'Sarvam API transport failed: {type(exc).__name__}; resume with the saved job state') from None
        if not response.ok:
            code = 'unknown'
            try:
                candidate = response.json().get('error', {}).get('code', '')
                if isinstance(candidate, str) and re.fullmatch(r'[a-z_]+', candidate):
                    code = candidate
            except (ValueError, AttributeError):
                pass
            raise RuntimeError(f'Sarvam API HTTP {response.status_code}: {code}')
        return response.json()

    def storage(self, method, url, data=None, azure=False):
        # Presigned storage URLs must never receive the API key or appear in logs.
        if urlsplit(url).scheme != 'https':
            raise ValueError('Expected an HTTPS storage URL')
        headers = {'Content-Type': 'audio/wav'} if method == 'PUT' else {}
        if azure:
            headers['x-ms-blob-type'] = 'BlockBlob'
        try:
            response = requests.request(method, url, data=data, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f'Storage transfer failed: {type(exc).__name__}') from None
        if not response.ok:
            raise RuntimeError(f'Storage transfer HTTP {response.status_code}')
        return response

    def process(self, records, parameters, checkpoint, audio_root, wait_seconds):
        identity = {'parameters': parameters, 'inputs': records}
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
            if state['identity'] != identity:
                raise ValueError('Checkpoint inputs/settings differ; use a new output directory')
            if state.get('results') is not None:
                return state['results']
        else:
            # Persist the identity before creating a paid job. A lost create response
            # is ambiguous: do not silently create another job on restart.
            state = {'identity': identity, 'creation_pending': True}
            write_json(checkpoint, state)
            created = self.api('POST', body={'job_parameters': parameters})
            state.update(job_id=created['job_id'], storage=created['storage_container_type'], creation_pending=False)
            write_json(checkpoint, state)
        if state.get('creation_pending') or not state.get('job_id'):
            raise RuntimeError('Job creation outcome is unknown; inspect the provider job before retrying this checkpoint')
        job_id = quote(state['job_id'], safe='')
        status = self.api('GET', f'/{job_id}/status')
        if status['job_state'] == 'Accepted':
            if not state.get('uploaded'):
                names = [r['upload_name'] for r in records]
                links = self.api('POST', '/upload-files', {'job_id': state['job_id'], 'files': names})
                for record in records:
                    url = links['upload_urls'][record['upload_name']]['file_url']
                    with (audio_root / record['relative_wav']).open('rb') as stream:
                        self.storage('PUT', url, data=stream, azure=links['storage_container_type'] in ('Azure', 'Azure_V1'))
                state['uploaded'] = True
                write_json(checkpoint, state)
            status = self.api('POST', f'/{job_id}/start')
        deadline = time.monotonic() + wait_seconds
        last_state = None
        while status['job_state'] not in TERMINAL:
            if time.monotonic() >= deadline:
                raise TimeoutError('ASR job still running; rerun the same command to resume it')
            if status['job_state'] != last_state:
                print(checkpoint.stem, status['job_state'], flush=True)
                last_state = status['job_state']
            time.sleep(5)
            status = self.api('GET', f'/{job_id}/status')
        state['job_state'] = status['job_state']
        state['results'] = self.collect(status, records, state['job_id'])
        write_json(checkpoint, state)
        return state['results']

    def collect(self, status, records, job_id):
        details = {}
        for detail in status.get('job_details', []):
            names = [x['file_name'] for x in detail.get('inputs', [])]
            if len(names) != 1 or names[0] in details:
                raise ValueError('Ambiguous input/output mapping in ASR response')
            details[names[0]] = detail
        results = []
        for record in records:
            detail = details.get(record['upload_name'], {})
            outputs = detail.get('outputs', [])
            if detail.get('state') != 'Success' or len(outputs) != 1:
                results.append({**record, 'status': 'failed', 'reason': detail.get('state', 'missing_file_result')})
                continue
            output_name = outputs[0]['file_name']
            links = self.api('POST', '/download-files', {'job_id': job_id, 'files': [output_name]})
            result = self.storage('GET', links['download_urls'][output_name]['file_url']).json()
            if not isinstance(result.get('transcript'), str):
                raise ValueError('ASR output has no transcript string')
            results.append({**record, 'status': 'ok', 'transcript': result['transcript'],
                            'request_id': result.get('request_id'), 'detected_language': result.get('language_code')})
        return results


def prepare_records(items, systems, audio_root, dataset_hash):
    records = []
    for system in systems:
        manifest = json.loads((audio_root / f'{system}_manifest.json').read_text())
        if manifest['sentences_sha256'] != dataset_hash:
            raise ValueError(f'{system}: dataset differs from generation manifest')
        generated = {row['id']: row for row in manifest['items']}
        for item in items:
            row = generated.get(item['id'], {})
            path = audio_root / system / (item['id'] + '.wav')
            if row.get('status') != 'ok' or not path.is_file():
                raise ValueError(f'Missing successful paired audio: {system}/{item["id"]}')
            digest = sha256(path)
            if row['wav_sha256'] != digest:
                raise ValueError(f'Audio hash differs: {system}/{item["id"]}')
            with wave.open(str(path), 'rb') as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 24000:
                    raise ValueError('High-quality track requires mono PCM16 at 24 kHz for both systems')
                seconds = wav.getnframes() / wav.getframerate()
                if not 0 < seconds <= 7200:
                    raise ValueError('Audio must be nonempty and within batch API duration limit')
            records.append({'system': system, 'id': item['id'], 'lang': item['lang'],
                            'language_code': item['sarvam'], 'upload_name': f'{system}__{item["id"]}.wav',
                            'relative_wav': f'{system}/{item["id"]}.wav', 'wav_sha256': digest,
                            'audio_seconds': seconds})
    return records



def load_reused_results(source, items, records, config):
    """Reuse raw ASR only when audio, input text, language and ASR settings match."""
    previous_config = json.loads((source / 'config.json').read_text())
    for key in ('model', 'mode', 'transport', 'diarization', 'sample_rate', 'systems'):
        if previous_config.get(key) != config.get(key):
            raise ValueError(f'Pilot ASR setting differs: {key}')
    rows = json.loads((source / 'transcripts.json').read_text())
    inputs = {(r['system'], r['id']): r for r in previous_config['inputs']}
    current = {(r['system'], r['id']): r for r in records}
    text = {r['id']: r['text'] for r in items}
    reused, seen = [], set()
    for row in rows:
        identity = (row['system'], row['id'])
        if identity in seen:
            raise ValueError('Duplicate ASR result in pilot')
        seen.add(identity)
        if identity not in current:
            continue
        target, original = current[identity], inputs.get(identity, {})
        if row.get('status') != 'ok':
            continue
        for key in ('wav_sha256', 'lang', 'language_code'):
            if row.get(key) != target.get(key) or original.get(key) != target.get(key):
                raise ValueError(f'Pilot audio/language differs for {identity}')
        if row.get('input_text') != text[row['id']]:
            raise ValueError(f'Pilot input text differs for {identity}')
        if not isinstance(row.get('transcript'), str):
            raise ValueError('Pilot ASR output has no transcript string')
        reused.append({**target, 'status':'ok', 'transcript':row['transcript'],
                       'request_id':row.get('request_id'), 'detected_language':row.get('detected_language'),
                       'reused_asr':True})
    return reused


def save_report(items, systems, results, config, output):
    indexed = {x['id']: x for x in items}
    rows = []
    for result in results:
        if result['status'] != 'ok':
            continue
        item = indexed[result['id']]
        reference = item.get('spoken_reference', item['text'])
        rows.append({**result, 'input_text': item['text'], 'reference': reference,
                     'hypothesis': result['transcript'], 'usecase': item.get('usecase'),
                     'has_digits': bool(re.search(r'\d', item['text'])),
                     **score_text(reference, result['transcript'])})
    paired = set.intersection(*({r['id'] for r in rows if r['system'] == system} for system in systems))
    complete = len(rows) == len(items) * len(systems)
    summary = {'config': config, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'expected_clips': len(items)*len(systems), 'scored_clips': len(rows),
               'coverage_complete': complete, 'matched_ids': sorted(paired),
               'failed': [r for r in results if r['status'] != 'ok'],
               'caveat': 'Written-text agreement through Sarvam ASR, not a human naturalness score. Numbers, code mixing, spelling and ASR normalization need review.',
               'systems': {}}
    for system in systems:
        group = [r for r in rows if r['system'] == system and r['id'] in paired]
        summary['systems'][system] = {'matched_only': aggregate(group),
            'by_language': {lang: aggregate([r for r in group if r['lang'] == lang]) for lang in sorted({r['lang'] for r in group})},
            'by_usecase': {cat: aggregate([r for r in group if r['usecase'] == cat]) for cat in sorted({r['usecase'] for r in group}, key=str)},
            'without_digits': aggregate([r for r in group if not r['has_digits']])}
    write_json(output / 'transcripts.json', rows)
    write_json(output / 'summary.json', summary)
    with (output / 'transcripts.tsv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['system','id','lang','usecase','WER_percent','CER_percent','reference','ASR_hypothesis'])
        for row in rows:
            writer.writerow([row['system'],row['id'],row['lang'],row['usecase'],100*row['wer'],100*row['cer'],row['reference'],row['hypothesis']])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sentences', type=Path, required=True)
    parser.add_argument('--audio-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--systems', nargs='+', default=['ours','sarvam'])
    parser.add_argument('--model', choices=['saaras:v3','saaras:v4'], default='saaras:v3')
    parser.add_argument('--mode', choices=['codemix','transcribe','verbatim'], default='codemix')
    parser.add_argument('--wait-seconds', type=int, default=900)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--reuse-asr', type=Path, help='Reuse verified raw transcripts from a completed smaller pilot')
    args = parser.parse_args()
    if len(args.systems) != 2 or len(set(args.systems)) != 2 or any(not re.fullmatch(r'[A-Za-z0-9_-]+',s) for s in args.systems):
        parser.error('Supply two distinct safe system names')
    if args.wait_seconds <= 0:
        parser.error('--wait-seconds must be positive')
    key = os.environ.get('SARVAM_API_KEY')
    if not args.dry_run and not key:
        parser.error('Set SARVAM_API_KEY locally before running Sarvam ASR')
    items = read_items(args.sentences)
    if any(x.get('eval_category','high_quality') != 'high_quality' for x in items):
        parser.error('This runner is for the high-quality track; keep 8 kHz telephony evaluation separate')
    digest = sha256(args.sentences)
    records = prepare_records(items, args.systems, args.audio_root, digest)
    config = {'sentences_sha256': digest, 'systems': args.systems, 'model': args.model,
              'mode': args.mode, 'transport': 'batch', 'diarization': False, 'sample_rate':24000,
              'inputs': records}
    reused = []
    if args.reuse_asr:
        if args.reuse_asr.resolve() == args.output.resolve():
            parser.error('Pilot ASR source and destination must be different')
        reused = load_reused_results(args.reuse_asr, items, records, config)
        config['reused_asr_source_sha256'] = sha256(args.reuse_asr / 'transcripts.json')
        config['reused_asr_config_sha256'] = sha256(args.reuse_asr / 'config.json')
    reused_ids = {(r['system'], r['id']) for r in reused}
    remaining = [r for r in records if (r['system'], r['id']) not in reused_ids]
    seconds = sum(r['audio_seconds'] for r in remaining)
    print(f'{len(records)} clips; {len(reused)} pilot ASR results reused; remaining input {seconds/3600:.3f} audio hours; upper-bound ASR estimate INR {seconds/3600*30:.2f} at 30/hour before existing job-cache reuse', flush=True)
    if args.dry_run:
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            parser.error('Saved ASR run settings/inputs differ; choose a new --output')
    elif any(args.output.iterdir()):
        parser.error('Output contains other files; choose a new directory')
    write_json(config_path, config)
    jobs = args.output / 'jobs'
    jobs.mkdir(exist_ok=True)
    grouped = defaultdict(list)
    for record in remaining:
        grouped[record['language_code']].append(record)
    client = SarvamBatch(key)
    results = list(reused)
    try:
        for lang, group in sorted(grouped.items()):
            group.sort(key=lambda r:(r['id'],r['system']))
            for offset in range(0, len(group), 20):
                batch = group[offset:offset+20]
                params = {'model': args.model, 'mode': args.mode, 'language_code': lang,
                          'with_diarization': False, 'with_timestamps': False, 'input_audio_codec':'wav'}
                checkpoint = jobs / f'{lang}_{offset//20:04d}.json'
                results.extend(client.process(batch, params, checkpoint, args.audio_root, args.wait_seconds))
                save_report(items, args.systems, results, config, args.output)
                print(lang, f'completed {len(results)}/{len(records)} files', flush=True)
    except Exception as exc:
        save_report(items, args.systems, results, config, args.output)
        print(f'Stopped: {exc}. Existing job checkpoints are retained; rerun the same command to resume.', flush=True)
        return 1
    summary = save_report(items, args.systems, results, config, args.output)
    for system, stats in summary['systems'].items():
        print(system, stats['matched_only'])
    return 0 if summary['coverage_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
