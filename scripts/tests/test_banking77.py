import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("prepare_banking77", Path(__file__).parents[1] / "prepare-banking77.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(ident, text, label="one"):
    return {"id": ident, "state": text, "label": label, "group": module.sha256(module.text_key(text).encode())}


class PreparationTests(unittest.TestCase):
    def test_preserves_test_and_removes_train_overlap(self):
        train, test, audit = module.clean_splits(
            [row("train1", "HELLO  WORLD"), row("train2", "unique"), row("train3", "Unique")],
            [row("test1", "hello world"), row("test2", "hello world")])
        self.assertEqual([r["id"] for r in train], ["train2"])
        self.assertEqual([r["id"] for r in test], ["test1"])
        self.assertEqual(len(audit), 3)

    def test_conflicting_annotations_are_not_silently_discarded(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            module.clean_splits([row("a", "same", "one"), row("b", "SAME", "two")], [])

    def test_split_is_deterministic_stratified_and_disjoint(self):
        rows = [row(str(i), f"example {i}", "one" if i < 10 else "two") for i in range(20)]
        a = module.validation_split(rows)
        b = module.validation_split(list(reversed(rows)))
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]), 16)
        self.assertEqual(len(a[1]), 4)
        self.assertEqual({r["label"] for r in a[1]}, {"one", "two"})
        self.assertFalse({r["group"] for r in a[0]} & {r["group"] for r in a[1]})

    def test_unicode_and_whitespace_deduplication(self):
        self.assertEqual(module.text_key(" ＨＥＬＬＯ\nworld "), module.text_key("hello world"))


if __name__ == "__main__":
    unittest.main()
