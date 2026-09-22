"""Offline regression tests: synthetic audio, mocked HTTP and mocked DNSMOS."""
import base64
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import generate
import mos_score
import report
import run
import voice_consistency
from score_sarvam import save_report


def invoke(module, args):
    with patch.object(sys, "argv", [module.__file__, *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
        return module.main()


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00\x10\x00" * 2400)
    return output.getvalue()


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "sentences.json"
        self.items = [{"id": "hi1", "lang": "hi", "sarvam": "hi-IN", "text": "नमस्ते दुनिया"},
                      {"id": "en1", "lang": "en", "sarvam": "en-IN", "text": "Hello world"}]
        common.write_json(self.dataset, {"items": self.items})
        self.inputs = []
        for system in ("ours", "sarvam"):
            (self.root / system).mkdir()
            entries = []
            for item in self.items:
                path = self.root / system / (item["id"] + ".wav")
                path.write_bytes(wav_bytes())
                entries.append({**item, "status": "ok", "wav_sha256": common.sha256(path)})
                self.inputs.append({"system": system, "id": item["id"], "lang": item["lang"],
                                    "wav_sha256": common.sha256(path)})
            common.write_json(self.root / (system + "_manifest.json"),
                              {"sentences_sha256": common.sha256(self.dataset), "items": entries})

    def mos_args(self):
        return ["--sentences", self.dataset, "--audio-root", self.root, "--output", self.root / "mos"]

    def score_mos(self, scorer=None):
        with patch.object(mos_score, "version", return_value="test"), patch.object(mos_score, "mos", scorer or Mock(return_value=(3.5, 3.8))):
            return invoke(mos_score, self.mos_args())

    def write_asr(self):
        config = {"sentences_sha256": common.sha256(self.dataset), "systems": ["ours", "sarvam"],
                  "model": "saaras:v3", "mode": "codemix", "inputs": self.inputs}
        text = {i["id"]: i["text"] for i in self.items}
        rows = [{**r, "status": "ok", "transcript": text[r["id"]]} for r in self.inputs]
        (self.root / "asr").mkdir()
        save_report(self.items, ["ours", "sarvam"], rows, config, self.root / "asr")

    def summaries(self):
        return (report.load(self.root / "asr/summary.json"), report.load(self.root / "mos/summary.json"),
                report.load(self.root / "mos/mos_scores.json"))

    def validate(self, asr, mos, scores):
        report.validate_coverage(self.items, common.sha256(self.dataset), ["ours", "sarvam"], asr, mos, scores)

    def test_failed_clip_is_excluded_from_both_sides_and_retried(self):
        def scorer(path):
            if path.parent.name == "sarvam" and path.stem == "en1":
                raise ValueError("synthetic failure")
            return 3.5, 3.8
        self.assertEqual(self.score_mos(scorer), 1)
        summary = report.load(self.root / "mos/summary.json")
        self.assertFalse(summary["coverage_complete"])
        self.assertEqual(summary["matched_ids"], ["hi1"])
        self.assertEqual([v["n"] for v in summary["systems"].values()], [1, 1])
        retry = Mock(return_value=(3.6, 3.9))
        self.assertEqual(self.score_mos(retry), 0)
        self.assertEqual(retry.call_count, 1)
        self.assertTrue(report.load(self.root / "mos/summary.json")["coverage_complete"])
        cached = Mock(side_effect=AssertionError("valid scores must be reused"))
        self.assertEqual(self.score_mos(cached), 0)
        cached.assert_not_called()

    def test_missing_and_changed_audio_cannot_reuse_cached_scores(self):
        self.score_mos()
        (self.root / "ours/hi1.wav").unlink()
        (self.root / "sarvam/en1.wav").write_bytes(b"changed")
        self.assertEqual(self.score_mos(), 1)
        result = report.load(self.root / "mos/summary.json")
        self.assertEqual(len(result["missing"]), 1)
        self.assertEqual(len(result["failed"]), 1)
        self.assertEqual(result["matched_ids"], [])

    def test_changed_dataset_is_rejected(self):
        self.score_mos()
        changed = copy.deepcopy(self.items)
        changed[0]["text"] = "नया पाठ"
        common.write_json(self.dataset, {"items": changed})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.score_mos()

    def test_report_accepts_complete_pairs_and_displays_counts(self):
        self.score_mos()
        self.write_asr()
        self.validate(*self.summaries())
        invoke(report, ["--run", self.root, "--sentences", self.dataset])
        result = (self.root / "REPORT.md").read_text()
        self.assertIn("2 prompts per system", result)
        self.assertIn("4 scored clips", result)

    def test_report_rejects_incomplete_coverage_different_ids_and_audio(self):
        self.score_mos()
        self.write_asr()
        cases = ["partial_asr", "unequal_mos", "wrong_ids", "wrong_audio", "wrong_language", "wrong_dataset"]
        for case in cases:
            with self.subTest(case=case):
                asr, mos, scores = self.summaries()
                if case == "partial_asr": asr["coverage_complete"] = False
                if case == "unequal_mos": mos["systems"]["sarvam"]["n"] = 1
                if case == "wrong_ids": mos["matched_ids"] = ["hi1", "other"]
                if case == "wrong_audio": scores["systems"]["sarvam"]["en1"]["wav_sha256"] = "different"
                if case == "wrong_language": mos["systems"]["sarvam"]["by_language"]["en"]["n"] = 0
                if case == "wrong_dataset": asr["config"]["sentences_sha256"] = "different"
                with self.assertRaises(ValueError): self.validate(asr, mos, scores)

    def test_report_rejects_legacy_unequal_mos_without_overwriting_report(self):
        self.write_asr()
        (self.root / "mos").mkdir()
        common.write_json(self.root / "mos/summary.json", {"ours": {"n": 22}, "sarvam": {"n": 18}})
        common.write_json(self.root / "mos/mos_scores.json", {})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            invoke(report, ["--run", self.root, "--sentences", self.dataset])
        self.assertEqual(error.exception.code, 1)
        self.assertFalse((self.root / "REPORT.md").exists())


class VoiceConsistencyTest(unittest.TestCase):
    def test_summary_is_leave_one_out_and_grouped_by_language_voice(self):
        records = [
            {"id": "hi1", "lang": "hi", "voice": "hi_female", "wav_sha256": "a"},
            {"id": "hi2", "lang": "hi", "voice": "hi_female", "wav_sha256": "b"},
            {"id": "en1", "lang": "en", "voice": "en_female", "wav_sha256": "c"},
            {"id": "en2", "lang": "en", "voice": "en_female", "wav_sha256": "d"},
        ]
        embeddings = {
            "hi1": np.array([1.0, 0.0]), "hi2": np.array([1.0, 0.0]),
            "en1": np.array([0.0, 1.0]), "en2": np.array([0.0, 1.0]),
        }
        result, rows = voice_consistency.summary(records, embeddings)
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["macro_mean_across_languages"], 1.0)
        self.assertEqual({row["id"] for row in rows}, {"hi1", "hi2", "en1", "en2"})


class PreprocessingTest(unittest.TestCase):
    def test_resampling_overshoot_is_clipped_before_dnsmos(self):
        import numpy as np
        fake = Mock(return_value={"p808_mos": 3.7, "ovrl_mos": 3.9})
        modules = {"librosa": SimpleNamespace(load=lambda *a, **k: (np.array([-1.014, .5, 1.014]), 16000)),
                   "speechmos": SimpleNamespace(dnsmos=SimpleNamespace(run=fake))}
        with patch.dict(sys.modules, modules):
            self.assertEqual(mos_score.mos("unused.wav"), (3.7, 3.9))
        np.testing.assert_array_equal(fake.call_args.args[0], np.array([-1, .5, 1], dtype=np.float32))

    def test_empty_nonfinite_audio_and_nonfinite_scores_are_rejected(self):
        import numpy as np
        for audio in (np.array([]), np.array([float("nan")]), np.array([float("inf")])):
            fake = Mock()
            with patch.dict(sys.modules, {"librosa": SimpleNamespace(load=lambda *a, **k: (audio, 16000)),
                 "speechmos": SimpleNamespace(dnsmos=SimpleNamespace(run=fake))}):
                with self.assertRaises(ValueError): mos_score.mos("unused.wav")
            fake.assert_not_called()
        with patch.dict(sys.modules, {"librosa": SimpleNamespace(load=lambda *a, **k: (np.array([0.1]), 16000)),
             "speechmos": SimpleNamespace(dnsmos=SimpleNamespace(run=lambda *a, **k: {"p808_mos": float("nan"), "ovrl_mos": 3.5}))}):
            with self.assertRaises(ValueError): mos_score.mos("unused.wav")


class AuthenticationTest(unittest.TestCase):
    def test_key_environment_file_conflict_and_transport(self):
        key = "test_" + "A" * 40
        self.assertEqual(common.tts_headers("https://tts.example/synthesize", {"TTS_API_KEY": key}),
                         {"Authorization": "Bearer " + key})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "private.key"
            path.write_text(key + "\n")
            self.assertEqual(common.tts_headers("http://127.0.0.1/synthesize", {"TTS_API_KEY_FILE": str(path)}),
                             {"Authorization": "Bearer " + key})
            with self.assertRaises(ValueError):
                common.tts_headers("https://tts.example", {"TTS_API_KEY": key, "TTS_API_KEY_FILE": str(path)})
            path.write_text("")
            with self.assertRaises(ValueError): common.tts_headers("https://tts.example", {"TTS_API_KEY_FILE": str(path)})
        for url in ("http://tts.example", "https://user:pass@tts.example", "https://tts.example?key=secret", "https://tts.example#secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                common.tts_headers(url, {"TTS_API_KEY": key})
        self.assertEqual(common.tts_headers("http://127.0.0.1:8080", {}), {})

    def test_generation_sends_keys_only_to_correct_provider_and_blocks_redirects(self):
        ours_key, sarvam_key = "ours_" + "A" * 40, "sarvam_" + "B" * 40
        for system, status in (("ours", 200), ("sarvam", 200), ("sarvam", 302)):
            with self.subTest(system=system, status=status), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                dataset = root / "sentences.json"
                common.write_json(dataset, {"items": [{"id": "en1", "text": "Hello world", "lang": "en", "sarvam": "en-IN"}]})
                response = Mock(status_code=status, content=wav_bytes())
                response.json.return_value = {"audios": [base64.b64encode(wav_bytes()).decode()]}
                with patch.dict(os.environ, {"TTS_API_KEY": ours_key, "SARVAM_API_KEY": sarvam_key}, clear=True), \
                     patch("requests.Session") as factory:
                    factory.return_value.__enter__.return_value.post.return_value = response
                    code = invoke(generate, ["--system", system, "--sentences", dataset, "--output", root,
                                             "--url", "https://tts.example"])
                    request = factory.return_value.__enter__.return_value.post
                    self.assertEqual(request.call_count, 1)
                    self.assertFalse(request.call_args.kwargs["allow_redirects"])
                    expected = {"Authorization": "Bearer " + ours_key} if system == "ours" else {"api-subscription-key": sarvam_key}
                    self.assertEqual(request.call_args.kwargs["headers"], expected)
                    self.assertEqual(code, 0 if status == 200 else 1)
                manifest = (root / f"{system}_manifest.json").read_text()
                self.assertNotIn(ours_key, manifest)
                self.assertNotIn(sarvam_key, manifest)

    def test_local_runner_stages_need_no_keys(self):
        for stage in ("mos", "consistency", "report"):
            with self.subTest(stage=stage), patch.dict(os.environ, {}, clear=True), patch.object(run, "run") as child:
                invoke(run, ["--stage", stage])
                self.assertEqual(child.call_count, 1)
                self.assertIn("--sentences", child.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
