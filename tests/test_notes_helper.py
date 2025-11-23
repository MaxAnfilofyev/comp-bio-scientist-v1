import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_scientist.utils.notes import ensure_note_files, read_note_file, write_note_file


class NotesHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.exp = self.base / "experiment_results"
        self.exp.mkdir(parents=True, exist_ok=True)
        self.prev_env = {k: os.environ.get(k) for k in ("AISC_BASE_FOLDER", "AISC_EXP_RESULTS")}
        os.environ["AISC_BASE_FOLDER"] = str(self.base)
        os.environ["AISC_EXP_RESULTS"] = str(self.exp)

    def tearDown(self) -> None:
        for k, v in self.prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_write_creates_canonical_and_shadow(self) -> None:
        result = write_note_file("hello", "pi_notes.md")
        canonical = Path(result["path"])
        shadow = self.base / "pi_notes.md"
        self.assertTrue(canonical.exists())
        self.assertEqual(canonical.read_text(), "hello")
        self.assertTrue(shadow.exists())
        if shadow.is_symlink():
            self.assertEqual(shadow.resolve(), canonical)
        else:
            self.assertEqual(shadow.read_text(), "hello")

    def test_merges_existing_shadow_copy(self) -> None:
        shadow = self.base / "user_inbox.md"
        shadow.write_text("root copy", encoding="utf-8")
        canonical, shadow_path = ensure_note_files("user_inbox.md")
        self.assertEqual(canonical.read_text(), "root copy")
        self.assertTrue(shadow_path.exists())
        if shadow_path.is_symlink():
            self.assertEqual(shadow_path.resolve(), canonical)
        else:
            self.assertEqual(shadow_path.read_text(), "root copy")

    def test_append_preserves_existing_content(self) -> None:
        write_note_file("line1", "pi_notes.md")
        write_note_file("line2", "pi_notes.md", append=True)
        note = read_note_file("pi_notes.md")
        self.assertEqual(note["content"], "line1\nline2")


if __name__ == "__main__":
    unittest.main()
