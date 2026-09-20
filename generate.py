#!/usr/bin/env python3
"""Generate the shared dataset with our TTS server or with Sarvam Bulbul.

Ours voice modes:
  lang_female  -> per-item "{lang}_female_default" (the intended per-language voice)
  en_default   -> "en_female_default" for every language (single-voice baseline)
  <preset>     -> that explicit preset for every item

Both systems write <output>/<system>/<id>.wav plus <output>/<system>_manifest.json
recording input text, voice, duration, WAV hash and status. Resumable.
"""
import argparse
import base64
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import time
import wave

from common import read_items, sha256, write_json, tts_headers, validate_tts_url

HERE = Path(__file__).resolve().parent


def resolve_voice(system, item, voice_mode):
    if system == "sarvam":
        return None  # provider speaker handled via --speaker
    if voice_mode == "lang_female":
        return f"{item['lang']}_female_default"
    if voice_mode == "en_default":
        return "en_female_default"
    return voice_mode  # explicit preset name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("ours", "sarvam"), required=True)
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Run directory; systems share it")
    parser.add_argument("--url", default=os.environ.get("OURS_TTS_URL", "http://127.0.0.1:8080"),
                        help="Our TTS server base URL (ours only)")
    parser.add_argument("--voice-mode", default=os.environ.get("OURS_VOICE_MODE", "lang_female"),
                        help="ours: lang_female | en_default | <preset name>")
    parser.add_argument("--speaker", default=os.environ.get("SARVAM_SPEAKER", "shubh"), help="Sarvam speaker")
    parser.add_argument("--model", default=os.environ.get("SARVAM_MODEL", "bulbul:v3"), help="Sarvam model")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--resume", action="store_true", help="Reuse verified successful files")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    items = read_items(args.sentences)
    key = os.environ.get("SARVAM_API_KEY")
    if args.system == "sarvam":
        if not key:
            parser.error("Set SARVAM_API_KEY in the environment")
        if any(not isinstance(x.get("sarvam"), str) for x in items):
            parser.error("Each item needs a Sarvam target code, e.g. te-IN")

    if args.system == "ours":
        url = args.url.rstrip("/") + "/synthesize"
        voice_label = args.voice_mode
    else:
        url = "https://api.sarvam.ai/text-to-speech"
        voice_label = args.speaker
    try:
        validate_tts_url(url)
        ours_headers = tts_headers(url) if args.system == "ours" else {}
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    directory = args.output / args.system
    manifest_path = args.output / (args.system + "_manifest.json")
    if (manifest_path.exists() or (directory.exists() and any(directory.iterdir()))) and not args.resume:
        parser.error("Existing results would be overwritten; choose a new --output or pass --resume")
    directory.mkdir(parents=True, exist_ok=True)

    import requests
    manifest = {"system": args.system, "created_utc": datetime.now(timezone.utc).isoformat(),
                "sentences_sha256": sha256(args.sentences), "endpoint": url,
                "voice": voice_label, "model": args.model if args.system == "sarvam" else "server_configured",
                "sample_rate_requested": 24000, "items": []}
    previous = {}
    if args.resume and manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        for k in ("system", "endpoint", "voice", "model", "sample_rate_requested", "sentences_sha256"):
            if saved.get(k) != manifest.get(k):
                parser.error(f"Existing run configuration differs: {k}")
        previous = {row["id"]: row for row in saved["items"]}
        manifest["created_utc"] = saved["created_utc"]
    manifest["items"] = [previous.get(item["id"], {**item, "status": "pending"}) for item in items]
    write_json(manifest_path, manifest)

    with requests.Session() as session:
        for index, item in enumerate(items):
            path = directory / (item["id"] + ".wav")
            existing = previous.get(item["id"], {})
            if existing.get("status") == "ok":
                if not path.is_file() or sha256(path) != existing.get("wav_sha256"):
                    raise ValueError(f"Saved audio is missing or modified: {item['id']}")
                print(item["id"], "reused", flush=True)
                continue
            voice = resolve_voice(args.system, item, args.voice_mode)
            started = time.monotonic()
            row = dict(item)
            response = None
            try:
                if args.system == "ours":
                    body = {"text": item["text"], "lang": item["lang"], "voice": voice}
                    headers = ours_headers
                else:
                    body = {"text": item["text"], "target_language_code": item["sarvam"],
                            "model": args.model, "speaker": args.speaker,
                            "speech_sample_rate": 24000, "output_audio_codec": "wav"}
                    headers = {"api-subscription-key": key}
                for attempt in range(4):
                    response = session.post(url, json=body, headers=headers, timeout=args.timeout, allow_redirects=False)
                    if response.status_code != 429 or attempt == 3:
                        break
                    time.sleep(10 * (attempt + 1))
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}; response body omitted")
                if args.system == "ours":
                    raw = response.content
                else:
                    audios = response.json()["audios"]
                    if len(audios) != 1:
                        raise RuntimeError("Expected one complete WAV")
                    raw = base64.b64decode(audios[0], validate=True)
                with wave.open(io.BytesIO(raw), "rb") as wav:
                    frames, rate = wav.getnframes(), wav.getframerate()
                    if not frames or rate != 24000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                        raise ValueError("Expected nonempty mono 24 kHz PCM16 WAV")
                    if len(wav.readframes(frames)) != frames * 2:
                        raise ValueError("Truncated WAV response")
                path.write_bytes(raw)
                row.update(status="ok", voice=voice or args.speaker, wav=f"{args.system}/{path.name}",
                           wav_sha256=sha256(path), audio_seconds=frames / rate,
                           elapsed_seconds=time.monotonic() - started)
                print(item["id"], voice or args.speaker, f"{frames/rate:.2f}s", flush=True)
            except Exception as exc:
                # Never persist request headers, keys or response bodies.
                row.update(status="failed", error_type=type(exc).__name__)
                if response is not None:
                    row["http_status"] = response.status_code
                print(item["id"], "FAILED", type(exc).__name__, flush=True)
            manifest["items"][index] = row
            write_json(manifest_path, manifest)
            if row["status"] != "ok" and response is not None and response.status_code in (401, 402, 403, 429):
                print("Stopped on auth/quota/rate-limit; resume after resolving it", flush=True)
                break
    failures = sum(x["status"] != "ok" for x in manifest["items"])
    print(f"Generated {len(items)-failures}/{len(items)} clips. Manifest: {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
