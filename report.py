#!/usr/bin/env python3
"""Publish a combined report only after validating complete ASR/MOS coverage."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from common import read_items, sha256


def load(path):
    return json.loads(Path(path).read_text())


def validate_coverage(items, dataset_hash, systems, asr, mos, scores):
    expected_ids = {r["id"] for r in items}
    expected_langs = Counter(r["lang"] for r in items)
    language = {r["id"]: r["lang"] for r in items}
    count = len(items)
    for label, summary in (("ASR", asr), ("MOS", mos)):
        if summary.get("coverage_complete") is not True or summary.get("failed") or summary.get("missing"):
            raise ValueError(f"{label} coverage is incomplete; rerun that stage")
        config = summary.get("config", {})
        if config.get("sentences_sha256") != dataset_hash or config.get("systems") != systems:
            raise ValueError(f"{label} dataset or systems differ from this report")
        ids = summary.get("matched_ids", [])
        if len(ids) != count or set(ids) != expected_ids:
            raise ValueError(f"{label} matched IDs differ from the full dataset")
        if summary.get("expected_clips") != count * len(systems) or summary.get("scored_clips") != count * len(systems):
            raise ValueError(f"{label} clip counts differ from the full dataset")
        if set(summary.get("systems", {})) != set(systems):
            raise ValueError(f"{label} is missing a requested system")
        for system in systems:
            stats = summary["systems"][system]
            n = stats.get("matched_only", {}).get("clips") if label == "ASR" else stats.get("n")
            by_language = stats.get("by_language", {})
            if n != count or set(by_language) != set(expected_langs):
                raise ValueError(f"{label}/{system} has incomplete language coverage")
            for lang, expected in expected_langs.items():
                actual = by_language[lang].get("clips" if label == "ASR" else "n")
                if actual != expected:
                    raise ValueError(f"{label}/{system}/{lang} has incomplete clip coverage")

    if scores.get("config") != mos["config"]:
        raise ValueError("MOS scores and summary use different configurations")
    inputs = asr["config"].get("inputs", [])
    by_input = {(r["system"], r["id"]): r for r in inputs}
    expected_pairs = {(s, i) for s in systems for i in expected_ids}
    if len(inputs) != len(expected_pairs) or set(by_input) != expected_pairs:
        raise ValueError("ASR input records do not cover every requested clip")
    for system in systems:
        per_id = scores.get("systems", {}).get(system, {})
        if set(per_id) != expected_ids:
            raise ValueError(f"MOS/{system} lacks verified per-clip scores; rerun MOS")
        for wid, result in per_id.items():
            original = by_input[system, wid]
            if (not result.get("wav_sha256") or result["wav_sha256"] != original.get("wav_sha256")
                    or result.get("lang") != language[wid] or original.get("lang") != language[wid]):
                raise ValueError(f"ASR and MOS used different audio/language: {system}/{wid}")
            if not all(isinstance(result.get(k), (int, float)) and math.isfinite(result[k])
                       for k in ("p808", "ovrl")):
                raise ValueError(f"Invalid MOS score: {system}/{wid}")


def validate_consistency(items, dataset_hash, systems, consistency, asr):
    expected_ids = {row["id"] for row in items}
    expected_langs = Counter(row["lang"] for row in items)
    if consistency.get("coverage_complete") is not True:
        raise ValueError("Voice-consistency coverage is incomplete")
    config = consistency.get("config", {})
    if config.get("sentences_sha256") != dataset_hash or config.get("systems") != systems:
        raise ValueError("Voice-consistency dataset or systems differ from this report")
    if consistency.get("expected_clips") != len(items) * len(systems) or consistency.get("scored_clips") != len(items) * len(systems):
        raise ValueError("Voice-consistency clip counts differ from the full dataset")
    inputs = {(row["system"], row["id"]): row for row in asr["config"].get("inputs", [])}
    for system in systems:
        stats = consistency.get("systems", {}).get(system, {})
        if stats.get("n") != len(items) or set(stats.get("per_language", {})) != set(expected_langs):
            raise ValueError(f"Voice consistency/{system} has incomplete coverage")
        scores = consistency.get("per_clip", {}).get(system, [])
        if len(scores) != len(items) or {row.get("id") for row in scores} != expected_ids:
            raise ValueError(f"Voice consistency/{system} lacks per-clip scores")
        for lang, count in expected_langs.items():
            if stats["per_language"][lang].get("n") != count:
                raise ValueError(f"Voice consistency/{system}/{lang} has incomplete coverage")
        for row in scores:
            original = inputs.get((system, row["id"]), {})
            if row.get("wav_sha256") != original.get("wav_sha256"):
                raise ValueError(f"Voice consistency used different audio: {system}/{row['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", default=["ours", "sarvam"])
    args = parser.parse_args()
    if len(args.systems) != 2 or len(set(args.systems)) != 2:
        parser.error("Supply two distinct system names")
    try:
        items = read_items(args.sentences)
        dataset_hash = sha256(args.sentences)
        asr_summary = load(args.run / "asr" / "summary.json")
        mos_summary = load(args.run / "mos" / "summary.json")
        scores = load(args.run / "mos" / "mos_scores.json")
        validate_coverage(items, dataset_hash, args.systems, asr_summary, mos_summary, scores)
        consistency_path = args.run / "voice_consistency" / "voice_consistency.json"
        consistency = load(consistency_path) if consistency_path.is_file() else None
        if consistency is not None:
            validate_consistency(items, dataset_hash, args.systems, consistency, asr_summary)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Report not written: {exc}\n")
    asr, mos = asr_summary["systems"], mos_summary["systems"]
    langs = sorted(Counter(r["lang"] for r in items))
    sysd = args.systems

    def cell(x):
        if isinstance(x, (int, float)):
            if not math.isfinite(x):
                raise ValueError("Cannot publish nonfinite metrics")
            return f"{x:.2f}"
        return str(x)

    out = ["# Pramiteeh TTS benchmark — ours vs Sarvam", "",
           f"Systems: {', '.join(sysd)}. Lower WER/CER is better; higher DNSMOS is better.", "",
           f"Verified coverage: **{len(items):,} prompts per system**, {len(langs)} languages; "
           f"**{len(items) * len(sysd):,} scored clips** in each of ASR and MOS. All IDs and audio hashes match.", "",
           f"Dataset SHA-256: `{dataset_hash}`.", "", "## Overall", "",
           "| Metric | " + " | ".join(sysd) + " |", "|" + "---|" * (len(sysd) + 1)]
    rows = [
        ("Corpus WER %", lambda s: asr[s]["matched_only"]["corpus_wer_percent"]),
        ("Corpus CER %", lambda s: asr[s]["matched_only"]["corpus_cer_percent"]),
        ("Sent-avg WER %", lambda s: asr[s]["matched_only"]["mean_sentence_wer_percent"]),
        ("Sent-avg CER %", lambda s: asr[s]["matched_only"]["mean_sentence_cer_percent"]),
        ("DNSMOS P.808", lambda s: mos[s]["p808"]),
        ("DNSMOS OVRL", lambda s: mos[s]["ovrl"]),
    ]
    if consistency is not None:
        rows.append(("Voice consistency", lambda s: consistency["systems"][s]["macro_mean_across_languages"]))
    for label, fn in rows:
        out.append(f"| {label} | " + " | ".join(cell(fn(s)) for s in sysd) + " |")
    if consistency is not None:
        out += ["", "## Per-language voice consistency (higher better)", "",
                "| Lang | " + " | ".join(sysd) + " |", "|" + "---|" * (len(sysd) + 1)]
        for lang in langs:
            out.append(f"| {lang} | " + " | ".join(cell(consistency["systems"][s]["per_language"][lang]["mean"]) for s in sysd) + " |")
    out += ["", "## Per-language Corpus WER % (lower better)", "",
            "| Lang | Clips per system | " + " | ".join(sysd) + " |", "|" + "---|" * (len(sysd) + 2)]
    for lang in langs:
        out.append(f"| {lang} | {mos[sysd[0]]['by_language'][lang]['n']} | " +
                   " | ".join(cell(asr[s]["by_language"][lang]["corpus_wer_percent"]) for s in sysd) + " |")
    out += ["", "## Per-language DNSMOS P.808 (higher better)", "",
            "| Lang | " + " | ".join(sysd) + " |", "|" + "---|" * (len(sysd) + 1)]
    for lang in langs:
        out.append(f"| {lang} | " + " | ".join(cell(mos[s]["by_language"][lang]["p808"]) for s in sysd) + " |")
    out += ["", "## Notes", "",
            f"- ASR: {asr_summary['config'].get('model')} / {asr_summary['config'].get('mode')}, applied to both systems including English.",
            "- WER/CER measure ASR agreement with the written reference; numbers, spelling and code-mixing can inflate errors.",
            "- DNSMOS uses finite mono 16 kHz audio clipped to [-1, 1] after resampling. It is a quality proxy, not a human MOS study.",
            "- Voice consistency, when present, is a within-language ECAPA speaker-embedding stability score over a fixed audio window. It is not a voice-cloning or listener-preference score.",
            "- All overall and per-language comparisons use identical prompt IDs across both systems.",
            "- The 22-prompt sample is a smoke test; use the full 1,265-prompt set for headline results.", ""]
    text = "\n".join(out)
    path = args.run / "REPORT.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    print(text)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
