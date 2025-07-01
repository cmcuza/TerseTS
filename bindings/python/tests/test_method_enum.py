import pathlib
import re
import unittest
from tersets import Method

class MethodEnumMatchTest(unittest.TestCase):
    def test_python_method_enum_matches_zig(self):
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        zig_file = repo_root / 'src' / 'tersets.zig'
        content = zig_file.read_text()
        match = re.search(r"pub const Method = enum\s*{([^}]*)}", content, re.MULTILINE)
        self.assertIsNotNone(match, 'Could not find Method enum in tersets.zig')
        body = match.group(1)
        zig_methods = []
        for line in body.splitlines():
            line = line.strip().rstrip(',')
            if line:
                zig_methods.append(line)
        python_methods = [member.name for member in Method]
        self.assertEqual(zig_methods, python_methods)
        for i, member in enumerate(Method):
            self.assertEqual(i, member.value)

