#!/usr/bin/env python3
"""One-command Indic TTS benchmark: ours vs Sarvam.

Pipeline (all resumable): generate ours -> generate Sarvam -> Sarvam ASR
transcripts + WER/CER -> DNSMOS naturalness -> combined REPORT.md.

Inputs (that is all you provide):
  SARVAM_API_KEY   Sarvam key (env or --sarvam-key-file) — used for Sarvam TTS + ASR.
  OURS_TTS_URL     Our TTS server base URL  (env or --ours-url).
  TTS_API_KEY_FILE Private inference key file (or TTS_API_KEY).

Local mos/report stages do not need API keys.

Example:
  export SARVAM_API_KEY=xxxx
  export OURS_TTS_URL=http://<host>:8080
  python run.py --dataset sample            # cheap 22-prompt smoke test
  python run.py --dataset full              # full 1265-prompt benchmark
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from common import tts_headers

HERE = Path(__file__).resolve().parent
DATASETS = {"sample": HERE / "data/sentences_sample.json", "full": HERE / "data/sentences_full.json"}


def run(cmd, env):
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n$ {printable}\n", flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode:
        sys.exit(f"Stage failed ({proc.returncode}); fix the cause and rerun — completed work is reused.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(DATASETS), default="sample")
    p.add_argument("--sentences", type=Path, help="Override with a custom dataset file")
    p.add_argument("--output", type=Path, help="Run directory (default: runs/<dataset>)")
    p.add_argument("--ours-url", default=os.environ.get("OURS_TTS_URL"))
    p.add_argument("--voice-mode", default=os.environ.get("OURS_VOICE_MODE", "lang_female"),
                   help="ours voice: lang_female | en_default | <preset>")
    p.add_argument("--sarvam-speaker", default=os.environ.get("SARVAM_SPEAKER", "shubh"))
    p.add_argument("--sarvam-model", default=os.environ.get("SARVAM_MODEL", "bulbul:v3"))
    p.add_argument("--sarvam-key-file", type=Path, help="File containing only the Sarvam key")
    p.add_argument("--tts-key-file", type=Path, help="Private inference key file; alternative to TTS_API_KEY_FILE")
    p.add_argument("--asr-model", default="saaras:v3")
    p.add_argument("--stage", choices=["all", "generate", "asr", "mos", "report"], default="all")
    args = p.parse_args()

    sentences = args.sentences or DATASETS[args.dataset]
    if not sentences.is_file():
        sys.exit(f"Dataset not found: {sentences}")
    output = args.output or (HERE / "runs" / args.dataset)
    py = [sys.executable, "-u"]

    env = os.environ.copy()
    needs_sarvam = args.stage in ("all", "generate", "asr")
    if needs_sarvam and args.sarvam_key_file:
        env["SARVAM_API_KEY"] = args.sarvam_key_file.read_text().strip()
    if needs_sarvam and not env.get("SARVAM_API_KEY"):
        sys.exit("Provide the Sarvam key via SARVAM_API_KEY or --sarvam-key-file")
    if args.stage in ("all", "generate"):
        if not args.ours_url:
            sys.exit("Provide our server via OURS_TTS_URL or --ours-url")
        if args.tts_key_file:
            env["TTS_API_KEY_FILE"] = str(args.tts_key_file)
        try:
            # Validate before printing commands or writing a generation manifest.
            tts_headers(args.ours_url, env)
        except (ValueError, OSError) as exc:
            sys.exit(str(exc))

    want = lambda s: args.stage in ("all", s)
    if want("generate"):
        run(py + [str(HERE / "generate.py"), "--system", "ours", "--sentences", str(sentences),
                  "--output", str(output), "--url", args.ours_url, "--voice-mode", args.voice_mode, "--resume"], env)
        run(py + [str(HERE / "generate.py"), "--system", "sarvam", "--sentences", str(sentences),
                  "--output", str(output), "--speaker", args.sarvam_speaker, "--model", args.sarvam_model, "--resume"], env)
    if want("asr"):
        run(py + [str(HERE / "score_sarvam.py"), "--sentences", str(sentences), "--audio-root", str(output),
                  "--systems", "ours", "sarvam", "--output", str(output / "asr"),
                  "--model", args.asr_model, "--mode", "codemix"], env)
    if want("mos"):
        run(py + [str(HERE / "mos_score.py"), "--sentences", str(sentences), "--audio-root", str(output),
                  "--systems", "ours", "sarvam", "--output", str(output / "mos")], env)
    if want("report"):
        run(py + [str(HERE / "report.py"), "--run", str(output), "--sentences", str(sentences),
                  "--systems", "ours", "sarvam"], env)
    print(f"\nDone. Results in {output}")


if __name__ == "__main__":
    raise SystemExit(main())
