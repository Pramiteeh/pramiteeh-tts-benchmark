# Pramiteeh TTS benchmark

Compare Pramiteeh and **Sarvam Bulbul v3** on the same prompts across **10 Indic
languages plus English**. The pipeline generates WAVs, transcribes both systems
with **Sarvam batch ASR**, scores **DNSMOS P.808/OVRL** locally, and writes a report
only after verifying complete, matching coverage.

## Full benchmark comparison

These are the completed **1,265-prompt results used in the blog**, with **115
prompts per language** across **10 Indic languages plus English**. Both systems
have **1,265 matched clips for ASR and DNSMOS** (2,530 scored clips per metric).
ASR completed on **17 September 2026**. The tables below use the full dataset;
the 22-prompt smoke test produces its own separate report.

**Settings:** Pramiteeh uses each language's `<lang>_female_default` preset;
Sarvam uses **Bulbul v3 / `shubh`**. Both are transcribed with **Saaras v3 /
`codemix`**, using explicit language hints. DNSMOS uses the corrected clipping
and equal-coverage procedure.

| Metric | Pramiteeh | Sarvam Bulbul v3 |
|---|---:|---:|
| Corpus WER % ↓ | 25.10 | 29.61 |
| Corpus CER % ↓ | 17.39 | 22.78 |
| Sentence-average WER % ↓ | 28.58 | 35.11 |
| Sentence-average CER % ↓ | 20.58 | 28.47 |
| DNSMOS P.808 ↑ | 3.69 | 4.11 |
| DNSMOS OVRL ↑ | 3.30 | 3.36 |

### Per-language comparison

Each row contains **115 clips per system**. WER is corpus-weighted; lower is
better. Higher DNSMOS P.808 is better.

| Language | Pramiteeh WER % ↓ | Sarvam WER % ↓ | Pramiteeh DNSMOS ↑ | Sarvam DNSMOS ↑ |
|---|---:|---:|---:|---:|
| Hindi | 20.76 | 28.19 | 3.61 | 4.12 |
| Bengali | 25.56 | 28.03 | 3.64 | 4.09 |
| Telugu | 27.17 | 35.97 | 3.72 | 4.10 |
| Tamil | 29.57 | 27.88 | 3.75 | 4.11 |
| Marathi | 21.12 | 25.73 | 3.82 | 4.10 |
| Gujarati | 35.86 | 43.97 | 3.62 | 4.12 |
| Kannada | 17.06 | 30.24 | 3.54 | 4.09 |
| Malayalam | 34.45 | 30.04 | 3.53 | 4.09 |
| Odia | 27.39 | 31.13 | 3.69 | 4.07 |
| Punjabi | 24.01 | 34.26 | 3.85 | 4.11 |
| English | 14.17 | 9.77 | 3.76 | 4.13 |

Pramiteeh has lower corpus WER in **8 of 11 languages** on this run. Sarvam has
higher DNSMOS P.808 in all 11. These measure ASR agreement and automatic audio
quality; neither is a human listening score. Numbers, spelling and code-mixing
can increase WER even when the spoken output is acceptable.

The [full-precision result snapshot](results/full_20260917.json) includes
coverage, scoring settings, per-language metrics and source-summary hashes.
These are retained measurements from the original run. Its Pramiteeh manifest
did not pin the server model/runtime commit, so rerunning against a changed
server or provider may produce different scores.

## Dataset source — Hugging Face

We downloaded the benchmark prompts from **Sarvam AI's public TTS benchmark**:

- **Hugging Face repository:** [sarvamai/tts-general-benchmark](https://huggingface.co/datasets/sarvamai/tts-general-benchmark)
- **Exact downloaded revision:** [c68a26f375d4b2ef310017310ef81405ca7695b0](https://huggingface.co/datasets/sarvamai/tts-general-benchmark/tree/c68a26f375d4b2ef310017310ef81405ca7695b0)
- **Original downloaded file:** [train-00000-of-00001.parquet](https://huggingface.co/datasets/sarvamai/tts-general-benchmark/resolve/c68a26f375d4b2ef310017310ef81405ca7695b0/data/train-00000-of-00001.parquet)

We use all **1,265 `high_quality` prompts**, **115 per language**, from that
revision. The separate **550 telephony prompts are excluded**. The converted
prompts are included in [data/sentences_full.json](data/sentences_full.json);
[data/sentences_sample.json](data/sentences_sample.json) contains 22 of those
prompts for a smoke test. Both TTS systems receive the same input text.

## Setup

Use Python 3.9+ in a virtual environment:

```bash
pip install -r requirements.txt
```

Pramiteeh inference runs on your existing TTS server. Sarvam generation and ASR
use its paid APIs. DNSMOS uses the models provided by `speechmos` locally.

## Run

From this repository:

```bash
export OURS_TTS_URL=https://serve.tts.pramiteeh.com
export TTS_API_KEY_FILE=/private/path/inference.key

python run.py --dataset sample --sarvam-key-file /private/path/sarvam.key
python run.py --dataset full --sarvam-key-file /private/path/sarvam.key
```

- `sample`: **22 prompts**, two per language; 44 generated WAVs across both systems.
- `full`: **1,265 prompts**, 115 per language; 2,530 generated WAVs across both systems.
- `TTS_API_KEY` can replace `TTS_API_KEY_FILE`; set only one. `--tts-key-file` is
  another way to select the file. It must contain the private **inference** key.
- `SARVAM_API_KEY` can replace `--sarvam-key-file`.
- An unauthenticated local development server can use
  `OURS_TTS_URL=http://127.0.0.1:8080` without an inference key.

Keys stay out of command output and result manifests. The inference key is sent
only to our server; the Sarvam key only to Sarvam. Remote requests carrying the
inference key require HTTPS. Generation does not follow redirects.

Local rescoring and report generation need **no API keys**:

```bash
python run.py --dataset sample --stage mos
python run.py --dataset sample --stage report
```

This reuses the existing WAVs and ASR transcripts. If upgrading from the earlier
MOS script, run these two commands to replace its unequal-coverage scores and
report. Legacy MOS output lacks verifiable per-clip hashes, so it is rescored.

## Server contract

Our server must implement `POST /synthesize`, accepting:

```json
{"text": "Hello world", "lang": "en", "voice": "en_female_default"}
```

The response must be a nonempty mono **24 kHz PCM16 WAV**. The default voice mode
constructs `<lang>_female_default` for each item; these presets must exist on the
server. The runner does not query `/voices` or guess a fallback voice.

`OURS_TTS_URL` is the inference server base URL. The website demo and its
`/api/demo/tts/stream` PCM endpoint are not this WAV interface. A website customer
key is not interchangeable with the private inference key.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `sample` | `sample` or `full` |
| `--sentences PATH` | — | Custom dataset JSON |
| `--output PATH` | `runs/<dataset>` | Run directory |
| `--ours-url` | `OURS_TTS_URL` | Inference server base URL |
| `--tts-key-file PATH` | `TTS_API_KEY_FILE` | Private inference key file |
| `--sarvam-key-file PATH` | `SARVAM_API_KEY` | Sarvam key file |
| `--voice-mode` | `lang_female` | Per-language female preset; `en_default` uses English female throughout; otherwise an explicit preset |
| `--sarvam-speaker` | `shubh` | Sarvam speaker |
| `--sarvam-model` | `bulbul:v3` | Sarvam TTS model |
| `--asr-model` | `saaras:v3` | Sarvam ASR model |
| `--stage` | `all` | `generate`, `asr`, `mos`, or `report` |

`OURS_VOICE_MODE`, `SARVAM_SPEAKER`, and `SARVAM_MODEL` also set those defaults.
For a custom dataset, use the same `--sentences` and `--output` for every stage.

## Results and resuming

```text
runs/sample/
  ours/*.wav                 ours_manifest.json
  sarvam/*.wav               sarvam_manifest.json
  asr/config.json            asr/jobs/*.json
  asr/transcripts.json        asr/transcripts.tsv
  asr/summary.json
  mos/mos_scores.json         mos/summary.json
  REPORT.md
```

Rerun the same command to resume. Generation verifies saved configuration and
WAV hashes. ASR retains provider job checkpoints; if job creation has an unknown
outcome, inspect the saved job before retrying to avoid duplicate charges.
MOS checkpoints each successful clip with its WAV hash and scorer configuration;
failed or missing clips are retried on rerun. Changing the dataset or scorer
package versions requires a new MOS output directory. Run only one writer per
output directory at a time.

MOS rejects empty/nonfinite audio and clips resampling overshoot to `[-1, 1]`
before scoring. Both systems' MOS aggregates use the intersection of successful
prompt IDs. Missing audio, hash mismatches, or scoring failures produce an
incomplete summary and a nonzero exit status.

The combined report additionally verifies:

- All expected IDs were scored by both ASR and MOS, with no failures.
- Dataset hashes and requested systems match.
- Every language has the expected clip count in both systems.
- ASR and MOS used the same WAV hash for every prompt/system pair.

A full report therefore requires **1,265 matched IDs and 2,530 scored clips in
each scoring stage**. Partial runs cannot produce a headline comparison. The
report includes coverage counts, dataset hash, corpus and sentence-average
WER/CER, and per-language results. An earlier report remains an earlier snapshot
if a later stage fails; publish only after the current report command succeeds.

## Dataset and interpretation

The included JSON is an exact conversion of the `high_quality` track of
[`sarvamai/tts-general-benchmark`](https://huggingface.co/datasets/sarvamai/tts-general-benchmark/tree/c68a26f375d4b2ef310017310ef81405ca7695b0),
revision `c68a26f375d4b2ef310017310ef81405ca7695b0`. The separate **550 telephony
prompts are excluded**. The frozen source parquet SHA-256 is
`244d5065fdbb6c040f22ef98ef4e153f8b06a0e840f7b66d4a1ecb6da77c3aa8`.
The data is for evaluation; retain the upstream attribution and usage terms.

Each item has a unique safe `id`, our `lang`, Sarvam's target code in `sarvam`,
and `text`. Optional `spoken_reference` changes the scoring reference without
changing the generated input. Odia uses `or` internally and `od-IN` for Sarvam.

- Both engines receive identical text. Ours defaults to each language's female
  preset; Sarvam defaults to `shubh`. Voice choice affects the comparison.
- ASR uses `saaras:v3`, `codemix`, and explicit language hints, including English.
  Both systems use the same normalization: NFC/lowercase, Unicode letters,
  marks/numbers and underscore retained, joiners removed, punctuation to spaces.
- WER/CER measure ASR agreement with the reference. Digits versus spoken number
  words, spelling, code-mixing and ASR mistakes can increase these scores.
- DNSMOS is an automatic quality proxy, not a human listening study. A 22-prompt
  smoke test does not establish performance on the full benchmark.
- HTTP elapsed time in generation manifests measures the complete response,
  including network and queue time. It is not time to first streamed audio.
- Keep model/runtime/voice versions fixed within a run. Manifests cannot prove
  an unchanged deployment behind the same server URL.

The standalone `score.py` also contains a legacy local IndicConformer CLI; the
supported runner uses only its text-scoring helpers, with cloud Sarvam ASR. The
local ASR CLI needs separate model assets and dependencies and is not used by
`run.py`.

## Validation and repository contents

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic WAVs and mocked HTTP/DNSMOS. They make no paid API requests.

Commit the Python source, `tests/`, `requirements.txt`, `data/`, `.gitignore`,
this README, and the reviewed `results/` snapshot supporting the comparison.
`CLAUDE.md` and `AGENTS.md` are optional assistant instructions, not dependencies.
Generated `runs/`, environments, caches and private keys are excluded from Git.
