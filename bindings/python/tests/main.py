import sys
import random
import unittest

from tersets import compress, decompress, Method, FunctionalParams, ErrorBoundType

uncompressed = [random.uniform(-100, 100) for _ in range(0, 100)]
params = FunctionalParams(error_bound_type=int(ErrorBoundType.ABS), error_bound=0.1)
compressed = compress(uncompressed, Method.SwingFilter, params)
decompressed = decompress(compressed)