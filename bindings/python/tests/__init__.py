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
        method = random.choice([Method.PoorMansCompressionMean, 
                                Method.PoorMansCompressionMidrange, 
                                Method.SwingFilter,
                                Method.ABCLinearApproximation,
                                Method.SwingFilterDisconnected,
                                Method.SlideFilter])
        
        compressed = compress(uncompressed, method, 0.0)
        decompressed = decompress(compressed)
        self.assertEqual(uncompressed, decompressed)


class MethodEnumMatchTest(unittest.TestCase):
    """Test that the Python Method enum matches the Zig and C Method enums."""
    def test_c_and_python_method_enum_matches_zig(self):
        """Test that the Python Method enum matches the Zig Method enum"""
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        zig_file = repo_root / 'src' / 'tersets.zig'
        # Read the Zig file and extract the Method enum.
        # This assumes the enum is defined as `pub const Method = enum { ... }`.
        content = zig_file.read_text()
        match = re.search(r"pub const Method = enum\s*{([^}]*)}", content, re.MULTILINE)
        # Ensure the Method enum is found in the Zig file.
        self.assertIsNotNone(match, 'Could not find Method enum in tersets.zig')
        body = match.group(1)
        zig_methods = []
        # Extract the method names from the Zig enum.
        for line in body.splitlines():
            line = line.strip().rstrip(',')
            if line:
                zig_methods.append(line)
        python_methods = [member.name for member in Method]
        # Ensure the methods match.
        self.assertEqual(zig_methods, python_methods)
        # Ensure the enum values match the expected indices.
        for i, member in enumerate(Method):
            self.assertEqual(i, member.value)

    def test_c_method_enum_matches_zig(self):
        """Test that the C Method enum matches the Zig Method enum"""
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        c_file = repo_root / 'bindings' / 'c' / 'tersets.h'
        c_content = c_file.read_text()
        # Extract the Method enum from the C header file.
        # This assumes the enum is defined as `enum Method { ... }`.
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
        # Extract the Method enum from the Zig file.
        # This assumes the enum is defined as `pub const Method = enum { ... }`.
        zig_match = re.search(r"pub const Method = enum\s*{([^}]*)}", zig_content, re.MULTILINE)
        self.assertIsNotNone(zig_match, 'Could not find Method enum in tersets.zig')
        zig_body = zig_match.group(1)
        zig_methods = []
        for line in zig_body.splitlines():
            line = line.strip().rstrip(',')
            if line:
                zig_methods.append(line)

        # Ensure the C methods match the Zig methods.
        self.assertEqual(zig_methods, c_methods)
        # Ensure the C enum values match the expected indices.
        self.assertEqual(list(range(len(c_methods))), c_values)