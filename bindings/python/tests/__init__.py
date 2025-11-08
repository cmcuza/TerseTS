# Copyright 2024 TerseTS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest
import pathlib
import struct
import math
import time
import re
from tersets import compress, decompress, Method


# Generate a random f64 value. This is equivalent to Zig's `random.int(u64)`
# followed by converting to `f64`.
def generate_random_f64():
    bits = random.getrandbits(64)  # equivalent to Zig's random.int(u64)
    return struct.unpack("d", struct.pack("Q", bits))[0]  # u64 → f64


# Check if a value is finite and not NaN.
def is_finite_and_real(x):
    return math.isfinite(x) and not math.isnan(x)


# Number of values to generate for each test.
TEST_VALUE_COUNT = 1000


class TerseTSPythonTest(unittest.TestCase):
    def test_compress_and_decompress_zero_error(self):
        """Test compressing and decompressing with zero error"""
        random.seed(time.time())
        count = 0
        uncompressed = []

        while count < TEST_VALUE_COUNT:
            random_value = generate_random_f64()
            if is_finite_and_real(random_value):
                count += 1
                uncompressed.append(random_value)

        # Randomly select a compression method for testing.
        # Only methods that support all f64 values are selected.
        method = random.choice(
            [
                Method.PoorMansCompressionMean,
                Method.PoorMansCompressionMidrange,
                Method.SwingFilter,
                Method.ABCLinearApproximation,
                Method.SwingFilterDisconnected,
                Method.SlideFilter,
            ]
        )

        compressed = compress(uncompressed, method, 0.0)
        decompressed = decompress(compressed)
        self.assertEqual(uncompressed, decompressed)


class MethodEnumMatchTest(unittest.TestCase):
    """Test that the Python Method enum matches the Zig and C Method enums."""

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    c_file = repo_root / "bindings" / "c" / "tersets.h"
    zig_file = repo_root / "src" / "tersets.zig"
    zig_enum_pattern = r"pub const Method = enum\s*{([^}]*)}"
    c_enum_pattern = r"enum\s+Method\s*{([^}]*)}"

    def _extract_enum_from_file(
        self, file_path: pathlib.Path, pattern: str
    ) -> list[str]:
        content = file_path.read_text()
        match = re.search(pattern, content, re.MULTILINE)
        self.assertIsNotNone(match, f"Could not find Method enum in {file_path.name}")
        body = match.group(1)
        enum_members = []
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            if line:
                if "=" in line:
                    name, _ = [part.strip() for part in line.split("=", 1)]
                    enum_members.append(name)
                else:
                    enum_members.append(line)
        return enum_members

    def _extract_enum_values_from_c(self, file_path: pathlib.Path) -> list[int]:
        content = file_path.read_text()
        match = re.search(self.c_enum_pattern, content, re.MULTILINE)
        self.assertIsNotNone(match, "Could not find enum Method in C header")
        body = match.group(1)
        values = []
        next_value = 0
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            if "=" in line:
                _, value = [part.strip() for part in line.split("=", 1)]
                next_value = int(value, 0)
            values.append(next_value)
            next_value += 1
        return values

    def test_c_and_python_enum_matches_zig(self):
        """Test that the C Method enum matches the Zig Method enum"""
        c_methods = self._extract_enum_from_file(self.c_file, self.c_enum_pattern)
        c_values = self._extract_enum_values_from_c(self.c_file)
        zig_methods = self._extract_enum_from_file(self.zig_file, self.zig_enum_pattern)
        python_methods = [member.name for member in Method]

        self.assertEqual(zig_methods, c_methods)
        self.assertEqual(list(range(len(c_methods))), c_values)
        self.assertEqual(zig_methods, python_methods)
        for i, member in enumerate(Method):
            self.assertEqual(i, member.value)
