"""Python bindings for the TerseTS library."""

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

import sys
import pathlib
import sysconfig
from typing import List
from enum import Enum, unique
from ctypes import cdll, Structure, c_ubyte, c_float, c_int, \
    c_double, c_size_t, POINTER, byref, string_at, cast

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    np = None
    _HAS_NUMPY = False

# Private function to load the library.
def __load_library():
    """Locates the correct library for this system and loads it."""

    if sys.platform == "win32":
        # SHLIB_SUFFIX is not set and .pyd is used by build.
        library_name = "tersets.pyd"
    else:
        library_name = "tersets" + sysconfig.get_config_var("SHLIB_SUFFIX")

    # Attempt to load the library installed as part of the Python package.
    library_folder = pathlib.Path(__file__).parent.parent.resolve()
    library_path = library_folder / library_name
    if library_path.exists():
        return cdll.LoadLibrary(str(library_path))

    # Attempt to load the library compiled in the development repository.
    repository_root = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
    library_folder = repository_root / "zig-out" / "lib"

    if sys.platform == "win32":
        # SHLIB_SUFFIX is not set and .dll is used by Zig.
        library_name = "tersets.dll"
        library_folder = repository_root / "zig-out" / "bin"
    elif sys.platform == "darwin":
        # SHLIB_SUFFIX is set to .so but macOS uses .dylib.
        library_name = "tersets.dylib"

    try:
        library_path = next(library_folder.glob("*" + library_name))
    except StopIteration:
        raise RuntimeError(f"Could not find TerseTS: looked '*{library_name}' in {library_folder}")
    return cdll.LoadLibrary(str(library_path))


# A global variable is used for the library so it is only initialized once and
# can be easily used by the public functions without users having to pass it.
__library = __load_library()

# Private Types.
class __UncompressedValues(Structure):
    _fields_ = [("data", POINTER(c_double)), ("len", c_size_t)]


class __CompressedValues(Structure):
    _fields_ = [("data", POINTER(c_ubyte)), ("len", c_size_t)]


class __Configuration(Structure):
    _fields_ = [("method", c_ubyte), ("error_bound", c_float)]

__library.compress.argtypes = [__UncompressedValues, POINTER(__CompressedValues), __Configuration]
__library.compress.restype  = c_int
__library.decompress.argtypes = [__CompressedValues, POINTER(__UncompressedValues)]
__library.decompress.restype  = c_int

# Mirror TerseTS Method Enum.
@unique
class Method(Enum):
    PoorMansCompressionMidrange = 0
    PoorMansCompressionMean = 1
    SwingFilter = 2
    SwingFilterDisconnected = 3
    SlideFilter = 4
    SimPiece = 5
    PiecewiseConstantHistogram = 6
    PiecewiseLinearHistogram = 7
    ABCLinearApproximation = 8
    VisvalingamWhyatt = 9
    SlidingWindow = 10
    BottomUp = 11
    MixPiece = 12
    BitPackedQuantization = 13
    RunLengthEncoding = 14


# ---- Public API ----
def compress(values, method, error_bound: float) -> bytes:
    """
    Compress a sequence of floats.

    Accepts:
      - numpy.ndarray[float64] (uses zero-copy fast path when NumPy is available)
      - list/tuple[float]      (fast ctypes path)
    Returns:
      - bytes (payload + trailing method byte)
    """
    if type(method) is not Method:
        available = ", ".join(m.name for m in Method)
        raise TypeError(f"'{method}' is not a valid TerseTS Method. Available: {available}")

    if _HAS_NUMPY and isinstance(values, np.ndarray):
        return _compress_numpy(values, method, error_bound)

    # Duck-typing for array-likes
    if _HAS_NUMPY and hasattr(values, "__array_interface__"):
        return _compress_numpy(np.ascontiguousarray(values, dtype=np.float64), method, error_bound)

    if isinstance(values, (list, tuple)):
        return _compress_list(list(values), method, error_bound)

    raise TypeError(
        "compress(values, ...): 'values' must be a numpy.ndarray[float64] "
        "or a list/tuple of floats."
    )


def decompress(values) -> List[float]:
    """
    Decompress a compressed stream to Python list[float].

    Accepts:
      - bytes / bytearray / memoryview (recommended)
      - list[int] (0..255 or signed) also accepted
    Returns:
      - list[float]
    """
    if _HAS_NUMPY:
        return _decompress_numpy(values)
    return _decompress_ctypes(values)


# --- Private functions ---
def _ensure_bytes(values):
    if isinstance(values, (bytes, bytearray, memoryview)):
        return bytes(values)
    if isinstance(values, list):
        return bytes((b & 0xFF for b in values))
    raise TypeError("compressed 'values' must be bytes-like or list[int]")


def _compress_numpy(values, method: Method, error_bound: float) -> bytes:
    if values.dtype != np.float64 or not values.flags["C_CONTIGUOUS"]:
        values = np.ascontiguousarray(values, dtype=np.float64)
    
    uncompressed_values = __UncompressedValues()
    
    uncompressed_values.data = values.ctypes.data_as(POINTER(c_double))
    uncompressed_values.len  = values.size
    
    compressed_values = __CompressedValues()
    
    configuration = __Configuration(method.value, error_bound)
    
    tersets_error = __library.compress(uncompressed_values, byref(compressed_values), configuration)
    
    if tersets_error != 0:
        raise RuntimeError(f"compress failed: {tersets_error}")
    
    return string_at(compressed_values.data, compressed_values.len)


def _compress_list(values: List[float], method: Method, error_bound: float) -> bytes:
    uncompressed_values = __UncompressedValues()
    
    uncompressed_values.data = (c_double * len(values))(*values)
    uncompressed_values.len  = len(values)
    
    compressed_values = __CompressedValues()
    
    configuration = __Configuration(method.value, error_bound)
    
    tersets_error = __library.compress(uncompressed_values, byref(compressed_values), configuration)
    
    if tersets_error != 0:
        raise RuntimeError(f"compress failed: {tersets_error}")
    
    return string_at(compressed_values.data, compressed_values.len)


def _decompress_numpy(values: bytes) -> List[float]:
    view = np.frombuffer(_ensure_bytes(values), dtype=np.uint8)  # zero-copy view.
    
    compressed_values  = __CompressedValues()
    
    compressed_values.data = view.ctypes.data_as(POINTER(c_ubyte))
    compressed_values.len  = view.size
    
    uncompressed_values = __UncompressedValues()
    
    tersets_error = __library.decompress(compressed_values, byref(uncompressed_values))
    
    if tersets_error != 0:
        raise RuntimeError(f"decompress failed: {tersets_error}")
    
    return np.ctypeslib.as_array(uncompressed_values.data, shape=(uncompressed_values.len,)).tolist()


def _decompress_ctypes(values: bytes) -> List[float]:
    buf = _ensure_bytes(values)
    
    compressed_values = __CompressedValues()
    # Need a writable buffer for ctypes without NumPy: make a copy.
    
    arr = (c_ubyte * len(buf)).from_buffer_copy(buf)
    
    compressed_values.data = cast(arr, POINTER(c_ubyte))
    compressed_values.len  = len(buf)
    
    uncompressed_values = __UncompressedValues()
    tersets_error = __library.decompress(compressed_values, byref(uncompressed_values))
    
    if tersets_error != 0:
        raise RuntimeError(f"decompress failed: {tersets_error}")
    
    return uncompressed_values.data[: uncompressed_values.len]  


