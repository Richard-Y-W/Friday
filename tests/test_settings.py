import unittest

from jarvis_research.settings import DEFAULT_SETTINGS, load_settings, set_setting


class SettingsTests(unittest.TestCase):
    def test_loads_defaults_and_persists_overrides(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".jarvis"

            defaults = load_settings(data_dir)
            updated = set_setting(data_dir, "research.limit", "250")
            reloaded = load_settings(data_dir)

            self.assertEqual(defaults["research"]["limit"], DEFAULT_SETTINGS["research"]["limit"])
            self.assertEqual(updated["research"]["limit"], 250)
            self.assertEqual(reloaded["research"]["limit"], 250)
            self.assertEqual(reloaded["research"]["deep_read_limit"], 10)
            self.assertEqual(reloaded["auto_label"]["provider"], "heuristic")
            self.assertEqual(reloaded["auto_label"]["model"], DEFAULT_SETTINGS["auto_label"]["model"])
            self.assertEqual(reloaded["corpus"]["paths"], "")
            self.assertEqual(reloaded["corpus"]["min_matches"], DEFAULT_SETTINGS["corpus"]["min_matches"])

            llm_updated = set_setting(data_dir, "auto_label.provider", "llm")
            model_updated = set_setting(data_dir, "auto_label.model", "gpt-test")
            corpus_updated = set_setting(data_dir, "corpus.paths", "/tmp/corpus-a.json:/tmp/corpus-b.json")

            self.assertEqual(llm_updated["auto_label"]["provider"], "llm")
            self.assertEqual(model_updated["auto_label"]["model"], "gpt-test")
            self.assertEqual(corpus_updated["corpus"]["paths"], "/tmp/corpus-a.json:/tmp/corpus-b.json")

    def test_rejects_unknown_setting_keys(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            with self.assertRaises(KeyError):
                set_setting(Path(tmp) / ".jarvis", "unknown.value", "1")


if __name__ == "__main__":
    unittest.main()
