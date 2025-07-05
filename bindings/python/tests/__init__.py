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
import random
import struct
import math
import time
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
