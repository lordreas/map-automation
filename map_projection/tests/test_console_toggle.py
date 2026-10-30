import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('console_toggle', Path(__file__).resolve().parents[1] / 'deploy/console_toggle.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConsoleChordTests(unittest.TestCase):
    def feed(self, events):
        chord = module.Chord()
        return [i for i, event in enumerate(events) if chord.event(*event)]

    def test_one_toggle_on_release_and_no_repeat(self):
        self.assertEqual(self.feed([(29,1),(56,1),(56,2),(29,0),(56,0)]), [4])

    def test_function_shortcuts_and_ctrl_alt_delete_do_not_toggle(self):
        for key in [60,61,111]:
            self.assertEqual(self.feed([(29,1),(56,1),(key,1),(key,0),(56,0),(29,0)]), [])

    def test_altgr_and_other_keys_do_not_toggle(self):
        self.assertEqual(self.feed([(29,1),(100,1),(100,0),(29,0)]), [])
        self.assertEqual(self.feed([(30,1),(29,1),(56,1),(56,0),(29,0),(30,0)]), [])

    def test_two_chords_toggle_twice(self):
        self.assertEqual(self.feed([(56,1),(97,1),(56,0),(97,0)] * 2), [3,7])
