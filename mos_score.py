#!/usr/bin/env python3
"""Resumable DNSMOS scoring with verified audio and equal clip coverage."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import statistics

from common import read_items, sha256, write_json

PREPROCESSING = "mono-16khz-finite-clip-unit-range-v1"


def mos(wav):
    import librosa
    import numpy as np
    from speechmos import dnsmos

    audio, _ = librosa.load(wav, sr=16000, mono=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("MOS requires nonempty finite audio")
    # Resampling can overshoot even when the original PCM16 WAV is valid.
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    result = dnsmos.run(audio, sr=16000)
    values = float(result["p808_mos"]), float(result["ovrl_mos"])
    if not all(math.isfinite(value) for value in values):
        raise ValueError("DNSMOS returned a nonfinite score")
    return values


def aggregate(scores, ids):
    return {"n": len(ids), **{
        metric: statistics.mean(scores[i][metric] for i in ids) if ids else None
        for metric in ("p808", "ovrl")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", default=["ours", "sarvam"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (len(args.systems) != 2 or len(set(args.systems)) != 2
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in args.systems)):
        parser.error("Supply two distinct safe system names")
    items = {it["id"]: it for it in read_items(args.sentences)}
    config = {"sentences_sha256": sha256(args.sentences), "systems": args.systems,
              "preprocessing": PREPROCESSING,
              "packages": {name: version(name) for name in ("speechmos", "librosa", "numpy", "onnxruntime")}}
    manifests = {}
    for system in args.systems:
        manifest = json.loads((args.audio_root / f"{system}_manifest.json").read_text())
        if manifest.get("sentences_sha256") != config["sentences_sha256"]:
            parser.error(f"{system}: dataset differs from generation manifest")
        entries = manifest["items"]
        manifests[system] = {r["id"]: r for r in entries}
        if len(manifests[system]) != len(entries) or set(manifests[system]) != set(items):
            parser.error(f"{system}: generation IDs differ from dataset")

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "mos_scores.json"
    cached = {s: {} for s in args.systems}
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if "config" in previous:
            if previous["config"] != config:
                parser.error("Saved MOS dataset/scorer settings differ; choose a new --output")
            cached = previous["systems"]
        else:
            print("Rescoring legacy MOS output: it has no verified per-clip checkpoint.", flush=True)
    scored = {s: {} for s in args.systems}
    failed, missing = [], []
    for system in args.systems:
        for index, (wid, item) in enumerate(items.items(), 1):
            case = {"system": system, "id": wid, "lang": item["lang"]}
            wav = args.audio_root / system / f"{wid}.wav"
            if not wav.is_file():
                missing.append(case)
                continue
            try:
                digest = sha256(wav)
                generated = manifests[system][wid]
                if generated.get("status") != "ok" or generated.get("wav_sha256") != digest:
                    raise ValueError("Audio does not match successful generation manifest")
                previous = cached.get(system, {}).get(wid, {})
                if (previous.get("wav_sha256") == digest and previous.get("lang") == item["lang"]
                        and all(isinstance(previous.get(k), (int, float))
                                and math.isfinite(previous[k]) for k in ("p808", "ovrl"))):
                    result = previous
                else:
                    p808, ovrl = mos(wav)
                    if not all(math.isfinite(x) for x in (p808, ovrl)):
                        raise ValueError("DNSMOS returned a nonfinite score")
                    result = {"lang": item["lang"], "wav_sha256": digest, "p808": p808, "ovrl": ovrl}
                scored[system][wid] = result
                cached[system][wid] = result
                write_json(checkpoint, {"config": config, "systems": cached})
            except Exception as exc:
                failed.append({**case, "error_type": type(exc).__name__})
                print(f"{system}/{wid}: MOS failed ({type(exc).__name__})", flush=True)
            if index % 200 == 0:
                print(f"{system}: checked {index}/{len(items)}", flush=True)
        print(f"{system}: scored {len(scored[system])}/{len(items)}", flush=True)

    common = sorted(set.intersection(*(set(scored[s]) for s in args.systems)))
    summary = {"config": config, "expected_clips": len(items) * len(args.systems),
               "scored_clips": sum(len(v) for v in scored.values()),
               "coverage_complete": len(common) == len(items), "matched_ids": common,
               "missing": missing, "failed": failed, "systems": {}}
    for system in args.systems:
        summary["systems"][system] = {
            **aggregate(scored[system], common), "available_clips": len(scored[system]),
            "by_language": {lang: aggregate(scored[system], [i for i in common if items[i]["lang"] == lang])
                            for lang in sorted({items[i]["lang"] for i in common})}}
    # Drop entries which failed revalidation; they must never reach a report.
    write_json(checkpoint, {"config": config, "systems": scored})
    write_json(args.output / "summary.json", summary)
    print(f"Matched coverage: {len(common)}/{len(items)} prompts per system. Saved: {args.output}/summary.json")
    return 0 if summary["coverage_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
