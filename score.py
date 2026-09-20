#!/usr/bin/env python3
"""Indic-safe ASR scoring, saved transcripts, coverage, and matched comparisons."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from common import normalize, read_items, sha256, write_json

HERE = Path(__file__).resolve().parent
ASR_ID = "ai4bharat/indic-conformer-600m-multilingual"
REVISION = "e9b71b369c048e2c6b634d4c131061c34e441179"


class ASR:
    def __init__(self, path, threads):
        import json
        import onnxruntime as ort
        import torch
        self.torch = torch
        torch.set_num_threads(threads)
        assets = path / "assets"
        self.preprocessor = torch.jit.load(str(assets / "preprocessor.ts"), map_location="cpu").eval()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.encoder = ort.InferenceSession(str(assets / "encoder.onnx"), options, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(assets / "ctc_decoder.onnx"), options, providers=["CPUExecutionProvider"])
        self.vocab = json.loads((assets / "vocab.json").read_text())
        self.masks = json.loads((assets / "language_masks.json").read_text())
        self.blank = json.loads((path / "config.json").read_text())["BLANK_ID"]
        self.identity = {"id": ASR_ID, "revision": REVISION if path.name == REVISION else "local_snapshot",
                         "decoding": "ctc", "provider": "CPUExecutionProvider", "threads": threads,
                         "torch": torch.__version__, "onnxruntime": ort.__version__,
                         "asset_sha256": {str(p.relative_to(path)): sha256(p) for p in [
                             path / "config.json", assets / "preprocessor.ts", assets / "encoder.onnx",
                             assets / "ctc_decoder.onnx", assets / "vocab.json", assets / "language_masks.json"]}}

    def transcribe(self, path, lang):
        import librosa
        torch = self.torch
        audio, _ = librosa.load(path, sr=16000, mono=True)
        signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            features, length = self.preprocessor(input_signal=signal, length=torch.tensor([signal.shape[-1]]))
            encoded, _ = self.encoder.run(["outputs", "encoded_lengths"],
                                          {"audio_signal": features.numpy(), "length": length.numpy()})
            logits = self.decoder.run(["logprobs"], {"encoder_output": encoded})[0]
            probabilities = torch.from_numpy(logits[:, :, self.masks[lang]]).log_softmax(dim=-1)
            ids = torch.unique_consecutive(torch.argmax(probabilities[0], dim=-1))
            return "".join(self.vocab[lang][int(i)] for i in ids if int(i) != self.blank).replace("▁", " ").strip()


def score_text(reference, hypothesis):
    import jiwer
    ref, hyp = normalize(reference), normalize(hypothesis)
    words, chars = jiwer.process_words(ref, hyp), jiwer.process_characters(ref, hyp)
    return {"normalized_reference": ref, "normalized_hypothesis": hyp,
            "wer": words.wer, "cer": chars.cer, "reference_words": len(ref.split()),
            "word_edits": words.substitutions + words.deletions + words.insertions,
            "reference_characters": len(ref),
            "character_edits": chars.substitutions + chars.deletions + chars.insertions}


def aggregate(rows):
    if not rows:
        return {"clips": 0}
    return {"clips": len(rows), "languages": len({x["lang"] for x in rows}),
            "mean_sentence_wer_percent": 100 * statistics.mean(x["wer"] for x in rows),
            "mean_sentence_cer_percent": 100 * statistics.mean(x["cer"] for x in rows),
            "corpus_wer_percent": 100 * sum(x["word_edits"] for x in rows) / sum(x["reference_words"] for x in rows),
            "corpus_cer_percent": 100 * sum(x["character_edits"] for x in rows) / sum(x["reference_characters"] for x in rows),
            "reference_words": sum(x["reference_words"] for x in rows),
            "word_edits": sum(x["word_edits"] for x in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", type=Path, default=HERE / "sentences.json")
    parser.add_argument("--audio-root", type=Path, required=True, help="Contains <system>/<id>.wav")
    parser.add_argument("--systems", nargs="+", default=["ours", "sarvam"])
    parser.add_argument("--output", type=Path, required=True, help="New scoring output directory")
    parser.add_argument("--asr-dir", type=Path, help="Existing ASR snapshot with assets/ and config.json")
    parser.add_argument("--download-asr", action="store_true", help="Explicitly allow downloading the pinned ASR snapshot")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import re
    if len(set(args.systems)) != len(args.systems) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in args.systems):
        parser.error("Systems must be unique safe directory names")
    if args.threads < 1:
        parser.error("--threads must be positive")
    items = read_items(args.sentences)
    dataset_hash = sha256(args.sentences)
    expected_audio = {}
    for system in args.systems:
        manifest = args.audio_root / (system + "_manifest.json")
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "sentences_sha256" in data:
                if data["sentences_sha256"] != dataset_hash:
                    parser.error(f"{system} was generated with a different dataset; use its original sentence file")
                expected_audio[system] = {row["id"]: row for row in data["items"]}
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new scoring output directory to preserve previous results")
    if args.asr_dir:
        snapshot = args.asr_dir
    else:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(ASR_ID, revision=REVISION, local_files_only=not args.download_asr,
                                          allow_patterns=["config.json", "assets/preprocessor.ts", "assets/encoder*",
                                                          "assets/ctc_decoder*", "assets/vocab.json", "assets/language_masks.json"]))
    print("Loading ASR on CPU; English is excluded if unsupported by the language mask", flush=True)
    asr = ASR(snapshot, args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, missing, failed, excluded = [], [], [], []
    for system in args.systems:
        for item in items:
            case = {"system": system, "id": item["id"], "lang": item["lang"]}
            if item["lang"] not in asr.masks:
                excluded.append({**case, "reason": "unsupported_by_asr"})
                continue
            path = args.audio_root / system / (item["id"] + ".wav")
            if not path.is_file():
                missing.append(case)
                continue
            try:
                wav_hash = sha256(path)
                if system in expected_audio:
                    expected = expected_audio[system].get(item["id"], {})
                    if expected.get("status") != "ok" or expected.get("wav_sha256") != wav_hash:
                        raise ValueError("WAV does not match the successful generation manifest")
                hyp = asr.transcribe(path, item["lang"])
                reference = item.get("spoken_reference", item["text"])
                row = {**case, "input_text": item["text"], "reference": reference, "hypothesis": hyp,
                       "wav_sha256": wav_hash, **score_text(reference, hyp)}
                rows.append(row)
                print(system, item["id"], f"WER {row['wer']*100:.2f}%", flush=True)
            except Exception as exc:
                failed.append({**case, "error": str(exc)[:300]})
                print(system, item["id"], "FAILED", type(exc).__name__, flush=True)
            write_json(args.output / "transcripts.json", rows)
    common = set.intersection(*({r["id"] for r in rows if r["system"] == s} for s in args.systems))
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "asr": asr.identity,
               "sentences_sha256": dataset_hash, "normalization": "NFC lowercase; preserve Unicode L/M/N and underscore; discard joiners; punctuation to spaces",
               "missing": missing, "failed": failed, "excluded": excluded,
               "coverage_complete_for_supported_languages": not missing and not failed,
               "matched_ids": sorted(common), "systems": {}}
    for system in args.systems:
        group = [r for r in rows if r["system"] == system]
        summary["systems"][system] = {"all_available": aggregate(group),
            "matched_only": aggregate([r for r in group if r["id"] in common]),
            "by_language": {lang: aggregate([r for r in group if r["lang"] == lang]) for lang in sorted({r["lang"] for r in group})}}
    write_json(args.output / "transcripts.json", rows)
    write_json(args.output / "summary.json", summary)
    with (args.output / "transcripts.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["system", "id", "lang", "WER_percent", "CER_percent", "reference", "ASR_hypothesis"])
        for r in rows:
            writer.writerow([r["system"], r["id"], r["lang"], 100*r["wer"], 100*r["cer"], r["reference"], r["hypothesis"]])
    for system, stats in summary["systems"].items():
        print(system, "matched:", stats["matched_only"])
    print(f"Missing: {len(missing)}; failed: {len(failed)}; unsupported: {len(excluded)}")
    return 1 if missing or failed or not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
