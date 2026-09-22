#!/usr/bin/env python3
"""Score within-language TTS voice consistency from generated benchmark WAVs.

Each clip is embedded with SpeechBrain's ECAPA-TDNN speaker encoder and compared
with the leave-one-out centroid of the other clips from the same system,
language and selected voice. This is a speaker-stability metric, not a measure
of naturalness, intelligibility, or voice-cloning fidelity.
"""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from common import read_items, sha256, write_json

SCORER = "speechbrain/spkrec-ecapa-voxceleb"
SCORER_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
TARGET_RATE = 16000


def audio_tensor(path, max_audio_seconds):
    """Read mono audio, resample to 16 kHz and apply the fixed scoring window."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError("empty, nonfinite, or silent WAV")
    if rate != TARGET_RATE:
        divisor = math.gcd(rate, TARGET_RATE)
        audio = resample_poly(audio, TARGET_RATE // divisor, rate // divisor).astype(np.float32)
    return audio[:round(max_audio_seconds * TARGET_RATE)]


def load_system(audio_root, system, expected_ids, dataset_hash):
    manifest_path = audio_root / f"{system}_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sentences_sha256") != dataset_hash:
        raise ValueError(f"{system}: generation manifest uses a different dataset")
    rows = {item["id"]: item for item in manifest.get("items", [])}
    if set(rows) != expected_ids:
        raise ValueError(f"{system}: generation IDs differ from the dataset")
    loaded = []
    for wid in sorted(expected_ids):
        item = rows[wid]
        path = audio_root / system / f"{wid}.wav"
        if item.get("status") != "ok" or not path.is_file():
            raise ValueError(f"{system}/{wid}: missing successful WAV")
        if item.get("wav_sha256") != sha256(path):
            raise ValueError(f"{system}/{wid}: WAV hash mismatch")
        loaded.append({"id": wid, "lang": item["lang"], "voice": item.get("voice", "unknown"), "path": path,
                       "wav_sha256": item["wav_sha256"]})
    return loaded, sha256(manifest_path)


def get_encoder(source, revision, cache_dir, device):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source=source, revision=revision, savedir=str(cache_dir),
                                         run_opts={"device": device})


def embed(records, encoder, device, batch_size, max_audio_seconds):
    """Return ID -> embedding. Long batches fall back to one clip at a time on OOM."""
    import torch
    ordered = sorted(records, key=lambda row: row["path"].stat().st_size)
    result = {}
    for start in range(0, len(ordered), batch_size):
        group = ordered[start:start + batch_size]
        audio = [torch.from_numpy(audio_tensor(row["path"], max_audio_seconds)) for row in group]
        lengths = torch.tensor([x.numel() for x in audio], dtype=torch.float32, device=device)
        batch = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True).to(device)
        lengths /= batch.shape[1]
        try:
            with torch.inference_mode():
                values = encoder.encode_batch(batch, lengths, normalize=False).reshape(len(group), -1).cpu().numpy()
        except torch.OutOfMemoryError:
            if not str(device).startswith("cuda"):
                raise
            torch.cuda.empty_cache()
            values = []
            for sample in audio:
                one = sample.unsqueeze(0).to(device)
                with torch.inference_mode():
                    values.append(encoder.encode_batch(one, torch.ones(1, device=device), normalize=False)
                                  .reshape(-1).cpu().numpy())
                torch.cuda.empty_cache()
            values = np.vstack(values)
        result.update({row["id"]: value for row, value in zip(group, values)})
        print(f"embedded {start + len(group)}/{len(ordered)}", flush=True)
    return result


def summary(records, embeddings):
    grouped = {}
    for row in records:
        grouped.setdefault((row["lang"], row["voice"]), []).append(row)
    values, per_language, rows = [], {}, []
    for (language, voice), group in sorted(grouped.items()):
        if len(group) < 2:
            raise ValueError(f"{language}/{voice}: need at least two clips")
        matrix = np.vstack([embeddings[row["id"]] for row in group])
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        scores = []
        for index, embedding in enumerate(matrix):
            centroid = matrix.sum(axis=0) - embedding
            centroid /= np.linalg.norm(centroid)
            score = float(np.dot(embedding, centroid))
            scores.append(score)
            rows.append({"id": group[index]["id"], "lang": language, "voice": voice,
                         "wav_sha256": group[index]["wav_sha256"], "speaker_cosine": score})
        per_language[language] = {"n": len(scores), "mean": statistics.mean(scores),
                                  "median": statistics.median(scores),
                                  "standard_deviation": statistics.stdev(scores)}
        values.extend(scores)
    return {"n": len(values), "macro_mean_across_languages": statistics.mean(v["mean"] for v in per_language.values()),
            "all_clips_mean": statistics.mean(values), "per_language": per_language}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", default=["ours", "sarvam"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-audio-seconds", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", help="Torch device; defaults to cuda:0 when available, otherwise cpu")
    parser.add_argument("--scorer", default=SCORER)
    parser.add_argument("--scorer-revision", default=SCORER_REVISION)
    args = parser.parse_args()
    if len(args.systems) != 2 or len(set(args.systems)) != 2:
        parser.error("Supply exactly two distinct systems")
    if args.max_audio_seconds <= 0 or args.batch_size < 1 or args.batch_size > 32:
        parser.error("audio duration and batch size must be positive; batch size is at most 32")

    import torch
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    items = read_items(args.sentences)
    dataset_hash = sha256(args.sentences)
    expected_ids = {item["id"] for item in items}
    loaded, manifest_hashes = {}, {}
    for system in args.systems:
        loaded[system], manifest_hashes[system] = load_system(args.audio_root, system, expected_ids, dataset_hash)

    args.output.mkdir(parents=True, exist_ok=True)
    config = {"metric": "within-language leave-one-out ECAPA-TDNN cosine similarity", "scorer": args.scorer,
              "scorer_revision": args.scorer_revision, "target_sample_rate": TARGET_RATE,
              "max_audio_seconds": args.max_audio_seconds, "systems": args.systems,
              "sentences_sha256": dataset_hash, "manifest_sha256": manifest_hashes, "device": device,
              "batch_size": args.batch_size,
              "packages": {name: version(name) for name in ("numpy", "soundfile", "scipy", "torch", "speechbrain")}}
    encoder = get_encoder(args.scorer, args.scorer_revision, args.output / "scorer", device)
    systems, per_clip = {}, {}
    for system in args.systems:
        embeddings = embed(loaded[system], encoder, device, args.batch_size, args.max_audio_seconds)
        systems[system], per_clip[system] = summary(loaded[system], embeddings)
    result = {"status": "complete", "config": config, "expected_clips": len(items) * len(args.systems),
              "scored_clips": sum(data["n"] for data in systems.values()), "coverage_complete": True,
              "systems": systems, "per_clip": per_clip,
              "interpretation": "Higher values mean a more stable speaker embedding across prompts within the same language and selected voice. This is not a naturalness, intelligibility, voice-cloning, or listener-preference measure."}
    write_json(args.output / "voice_consistency.json", result)
    print(f"Saved {args.output / 'voice_consistency.json'}")


if __name__ == "__main__":
    main()
