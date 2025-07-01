import pathlib
import re
import unittest

class CMethodEnumMatchTest(unittest.TestCase):
    def test_c_method_enum_matches_zig(self):
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        c_file = repo_root / 'bindings' / 'c' / 'tersets.h'
        c_content = c_file.read_text()
        c_match = re.search(r"enum\s+Method\s*{([^}]*)}", c_content, re.MULTILINE)
        self.assertIsNotNone(c_match, 'Could not find enum Method in tersets.h')
        c_body = c_match.group(1)
        c_methods = []
        c_values = []
        for line in c_body.splitlines():
            line = line.strip().rstrip(',')
            if not line:
                continue
            if '=' in line:
                name, value = [part.strip() for part in line.split('=', 1)]
                c_methods.append(name)
                c_values.append(int(value, 0))
            else:
                c_methods.append(line)
                c_values.append(None)

        zig_file = repo_root / 'src' / 'tersets.zig'
        zig_content = zig_file.read_text()
        zig_match = re.search(r"pub const Method = enum\s*{([^}]*)}", zig_content, re.MULTILINE)
        self.assertIsNotNone(zig_match, 'Could not find Method enum in tersets.zig')
        zig_body = zig_match.group(1)
        zig_methods = []
        for line in zig_body.splitlines():
            line = line.strip().rstrip(',')
            if line:
                zig_methods.append(line)

        self.assertEqual(zig_methods, c_methods)
        self.assertEqual(list(range(len(c_methods))), c_values)

