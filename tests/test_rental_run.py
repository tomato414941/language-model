import hashlib
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

from pretrain.recipe import data_fingerprint
from pretrain.rental_run import GPU_TESTS, write_run

SETTINGS = Path("configs/pretrain-v7-rental.toml")
CORPORA = ("corpus-patch", "corpus-stable", "corpus-late", "corpus-decay")


class RentalRunTests(unittest.TestCase):
    """v7 written as a run for the rental tool, from data built elsewhere, as the preparation leaves it."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.built = self.root / "built"
        fingerprint = data_fingerprint("configs/pretrain-v7.toml")
        self.write(self.built / "tokenizer/manifest.json", {
            "data_fingerprint": fingerprint, "tokenizer_sha256": "t", "benchmarks_sha256": "b"})
        for name in CORPORA:
            self.write(self.built / name / "manifest.json", {
                "dtype": "<u2", "data_fingerprint": fingerprint, "tokenizer_sha256": "t",
                "files": {f"{split}_{language}.bin": {"tokens": 10, "sha256": f"{name}-{split}-{language}"}
                          for split in ("train", "validation") for language in ("ja", "en")}})
            (self.built / name / "tokenizer.json").write_text("{}")
        self.write(self.built / "instructions/manifest.json", {
            "kind": "instructions", "dtype": "<u4", "context": 8192, "tokenizer_sha256": "t",
            "benchmarks_sha256": "b", "files": {"train_ja.bin": {"rows": 2, "sha256": "i"},
                                                "train_ja_mask.bin": {"rows": 2, "sha256": "m"}}})
        shutil.copy("configs/model-v7.toml", self.built / "sft.toml")
        self.questions = self.root / "questions"
        (self.questions / "datasets").mkdir(parents=True)
        (self.questions / "datasets/hellaswag.json").write_text("{}")

    def tearDown(self):
        self.directory.cleanup()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def run_spec(self, name="run", settings=SETTINGS):
        write_run(settings, self.built, self.root / name, "bucket", self.questions)
        return json.loads((self.root / name / "spec.json").read_text())

    def test_turns_the_experiment_into_steps_each_reading_only_files_or_what_earlier_steps_left(self):
        spec = self.run_spec()
        self.assertEqual([step["name"] for step in spec["steps"]], [
            "pretrain-v7-patch", "pretrain-v7-stable", "pretrain-v7-late", "pretrain-v7-decay",
            "benchmark-pretrain-v7-decay", "average-pretrain-v7-decay", "benchmark-pretrain-v7-decay-averaged",
            "instructions-v7", "benchmark-instructions-v7", "ruler-instructions-v7"])
        available = {item["path"] for item in spec["files"]}
        for step in spec["steps"]:
            for path in step.get("inputs", []):
                self.assertTrue(path in available or any(path.startswith(f"{made}/") or path == made
                                                         for made in available), (step["name"], path))
            available.update(step.get("outputs", []))

    def test_has_each_machine_pass_the_tests_only_a_cuda_gpu_runs_as_its_check(self):
        spec = self.run_spec()
        self.assertEqual(spec["check"], [[".venv/bin/python", "-m", "unittest", *GPU_TESTS]])
        loader = unittest.TestLoader()
        self.assertEqual(loader.loadTestsFromNames(GPU_TESTS).countTestCases(), len(GPU_TESTS))
        self.assertEqual(loader.errors, [])

    def test_refuses_settings_whose_rates_do_not_name_the_trainings(self):
        recipe = (SETTINGS.parent / "pretrain-v7.toml").resolve()
        changed = self.root / "rental.toml"
        changed.write_text(SETTINGS.read_text().replace('recipe = "pretrain-v7.toml"', f'recipe = "{recipe}"')
                           .replace("pretrain-v7-late = 39_500\n", ""))
        with self.assertRaisesRegex(ValueError, "the trainings are"):
            self.run_spec(settings=changed)

    def test_reads_every_corpus_the_trainings_name_the_late_one_with_them(self):
        spec = self.run_spec()
        read = {Path(item["path"]).parent.name for item in spec["files"]}
        self.assertEqual(read, {*CORPORA, "instructions"})
        late = next(step for step in spec["steps"] if step["name"] == "pretrain-v7-late")
        self.assertIn("data/pretrain-v7/corpus-late/train_ja.bin", late["inputs"])
        self.assertEqual(next(item for item in spec["files"] if item["path"].endswith("instructions/train_ja.bin"))
                         ["bytes"], 2 * 8193 * 4)

    def test_keeps_the_decay_snapshots_for_the_average_and_no_other_phase_snapshots(self):
        steps = {step["name"]: step for step in self.run_spec()["steps"]}
        self.assertIn("runs/pretrain-v7-decay/snapshots", steps["pretrain-v7-decay"]["outputs"])
        self.assertNotIn("runs/pretrain-v7-late/snapshots", steps["pretrain-v7-late"]["outputs"])
        self.assertIn("runs/pretrain-v7-decay/snapshots", steps["average-pretrain-v7-decay"]["inputs"])

    def test_gives_the_benchmarks_their_questions_and_the_trainings_their_expected_rates(self):
        steps = {step["name"]: step for step in self.run_spec()["steps"]}
        self.assertEqual(steps["benchmark-instructions-v7"]["environment"]["HF_DATASETS_OFFLINE"], "1")
        self.assertIn("--batch-tokens", steps["benchmark-instructions-v7"]["command"])
        self.assertEqual((steps["pretrain-v7-patch"]["rate"], steps["pretrain-v7-stable"]["rate"],
                          steps["instructions-v7"]["rate"]), (162_500, 39_500, 32_400))
        # The token phases share one schedule: each counts the updates it adds to the last.
        self.assertEqual(steps["pretrain-v7-stable"]["units"] + steps["pretrain-v7-late"]["units"]
                         + steps["pretrain-v7-decay"]["units"], steps["pretrain-v7-stable"]["units"] * 3000 // 1200)

    def test_refuses_data_built_for_another_recipe(self):
        manifest = self.built / "corpus-late/manifest.json"
        self.write(manifest, {**json.loads(manifest.read_text()), "data_fingerprint": "another"})
        with self.assertRaisesRegex(ValueError, "corpus-late belongs to another recipe"):
            self.run_spec()

    def test_writes_the_same_code_for_the_same_commit_and_data(self):
        self.run_spec("first")
        self.run_spec("second")
        first, second = (hashlib.sha256((self.root / name / "code.tar.gz").read_bytes()).hexdigest()
                         for name in ("first", "second"))
        self.assertEqual(first, second)
        with tarfile.open(self.root / "first/code.tar.gz") as bundle:
            names = set(bundle.getnames())
        self.assertLessEqual({"SOURCE_COMMIT", "data/pretrain-v7/corpus-late/manifest.json",
                              "data/pretrain-v7/sft.toml", "huggingface/datasets/hellaswag.json",
                              "pretrain/training.py"}, names)


if __name__ == "__main__":
    unittest.main()
