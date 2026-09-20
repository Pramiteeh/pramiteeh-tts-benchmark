# Pramiteeh TTS benchmark — ours vs Sarvam

Systems: ours, sarvam. Lower WER/CER is better; higher DNSMOS is better.

Verified coverage: **22 prompts per system**, 11 languages; **44 scored clips** in each of ASR and MOS. All IDs and audio hashes match.

Dataset SHA-256: `34cec1e6e90541b93b89295280a1661ba02fdf304e978167fac54e1758254976`.

## Overall

| Metric | ours | sarvam |
|---|---|---|
| Corpus WER % | 38.14 | 48.45 |
| Corpus CER % | 31.12 | 42.48 |
| Sent-avg WER % | 37.51 | 47.94 |
| Sent-avg CER % | 31.30 | 41.76 |
| DNSMOS P.808 | 3.54 | 4.06 |
| DNSMOS OVRL | 3.25 | 3.30 |

## Per-language Corpus WER % (lower better)

| Lang | Clips per system | ours | sarvam |
|---|---|---|---|
| bn | 2 | 42.31 | 73.08 |
| en | 2 | 11.54 | 11.54 |
| gu | 2 | 21.88 | 59.38 |
| hi | 2 | 46.67 | 36.67 |
| kn | 2 | 23.08 | 30.77 |
| ml | 2 | 50.00 | 72.73 |
| mr | 2 | 29.63 | 55.56 |
| or | 2 | 62.96 | 51.85 |
| pa | 2 | 45.16 | 51.61 |
| ta | 2 | 50.00 | 22.73 |
| te | 2 | 40.91 | 68.18 |

## Per-language DNSMOS P.808 (higher better)

| Lang | ours | sarvam |
|---|---|---|
| bn | 3.38 | 4.13 |
| en | 3.71 | 4.05 |
| gu | 3.35 | 4.04 |
| hi | 3.44 | 3.87 |
| kn | 3.55 | 4.04 |
| ml | 3.41 | 3.99 |
| mr | 3.80 | 4.21 |
| or | 3.44 | 4.06 |
| pa | 3.76 | 4.06 |
| ta | 3.59 | 4.03 |
| te | 3.56 | 4.20 |

## Notes

- ASR: saaras:v3 / codemix, applied to both systems including English.
- WER/CER measure ASR agreement with the written reference; numbers, spelling and code-mixing can inflate errors.
- DNSMOS uses finite mono 16 kHz audio clipped to [-1, 1] after resampling. It is a quality proxy, not a human MOS study.
- All overall and per-language comparisons use identical prompt IDs across both systems.
- The 22-prompt sample is a smoke test; use the full 1,265-prompt set for headline results.
