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
import json
from typing import List
from enum import Enum, unique
from ctypes import (
    cdll,
    Structure,
    c_ubyte,
    c_int,
    c_char_p,
    c_uint8,
    c_double,
    c_size_t,
    POINTER,
    byref,
    string_at,
    cast,
    
)

try:
    import numpy as np

    _INSTALLED_NUMPY = True
except Exception:
    np = None
    _INSTALLED_NUMPY = False


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
        raise RuntimeError(
            f"Could not find TerseTS: looked '*{library_name}' in {library_folder}"
        )
    return cdll.LoadLibrary(str(library_path))


# A global variable is used for the library so it is only initialized once and
# can be easily used by the public functions without users having to pass it.
__library = __load_library()


# Private Types.
class __UncompressedValues(Structure):
    _fields_ = [("data", POINTER(c_double)), ("len", c_size_t)]


class __CompressedValues(Structure):
    _fields_ = [("data", POINTER(c_ubyte)), ("len", c_size_t)]


# Declare function signatures (safer; avoids silent arg mismatch).
__library.compress.argtypes = [
    __UncompressedValues,
    POINTER(__CompressedValues), 
    c_uint8, 
    c_char_p,
]
__library.compress.restype = c_int
__library.decompress.argtypes = [__CompressedValues, POINTER(__UncompressedValues)]
__library.decompress.restype = c_int


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
    NonLinearApproximation = 15


# Public API. 
def compress(values, method, configuration) -> bytes:
    """Compress a sequence of float64 values with a selected TerseTS method.

    This function uses a zero-copy fast path when `values` is a NumPy
    `ndarray` of dtype `float64` and C-contiguous. Otherwise, it falls back to
    a ctypes copy for Python lists/tuples of floats.

    Args:
      values: Either a `numpy.ndarray` of dtype `float64` (C-contiguous) or a
        `list`/`tuple` of floats.
      method: A `Method` enum value selecting the compressor.
      configuration: Either a `dict` (will be JSON-encoded) or a JSON `str`.

    Returns:
      Compressed payload as `bytes` (payload only; the trailing method byte is
      handled by the native library).

    Raises:
      TypeError: If `method` is not a `Method` or `values`/`configuration` have
        unsupported types.
      RuntimeError: If the native `compress` call returns a non-zero error code.
    """
    # Validate compressor method type.
    if not isinstance(method, Method):
        available = ", ".join(m.name for m in Method)
        raise TypeError(
            f"{method!r} is not a valid TerseTS Method. Available: {available}"
        )

    # Prepare an __UncompressedValues view and keep a Python reference alive during the call.
    # The reference to the source buffer (NumPy array or ctypes array) must stay alive so the
    # garbage collector does not free the underlying memory while the native code reads from it.
    uncompressed_values = __UncompressedValues()

    if _INSTALLED_NUMPY and isinstance(values, np.ndarray):
        # Ensure dtype and memory layout are correct for zero-copy access.
        # If `values` is already float64 and C-contiguous, no copy is made.
        # Otherwise, np.ascontiguousarray() converts it (potentially copying)
        # so we can safely pass its data pointer directly to the native layer.
        if values.dtype != np.float64 or not values.flags["C_CONTIGUOUS"]:
            values = np.ascontiguousarray(values, dtype=np.float64)
        uncompressed_values.data = values.ctypes.data_as(POINTER(c_double))
        uncompressed_values.len = values.size
    elif isinstance(values, (list, tuple)):
        # Build a contiguous C array of f64 values from the Python sequence.
        # Python lists/tuples store references to float objects, not the raw
        # float64 values themselves, so the data is not contiguous in memory.
        # We copy each element into a new C double[] buffer to ensure the
        # native layer can read a continuous block of numeric values.
        buf = (c_double * len(values))(*values)
        uncompressed_values.data = buf
        uncompressed_values.len = len(values)
    else:
        # Invalid input type.
        raise TypeError(
            "values must be a numpy.ndarray[float64] that is C-contiguous, or a list/tuple of floats."
        )

    # Prepare compressed values buffer.
    compressed_values = __CompressedValues()

    # Accept configuration as dict (JSON-encode) or as JSON string.
    if isinstance(configuration, dict):
        json_configuration = json.dumps(configuration).encode("utf-8")
    elif isinstance(configuration, str):
        json_configuration = configuration.encode("utf-8")
    else:
        raise TypeError("configuration must be a dict, str, or bytes (JSON)")

    try:
        # Call native library.
        tersets_error = __library.compress(
            uncompressed_values,
            byref(compressed_values),
            c_uint8(method.value),
            c_char_p(json_configuration)
        )
        if tersets_error != 0:
            raise RuntimeError(f"compress failed: {tersets_error}")

        # Copy the compressed bytes into Python-owned memory (safe to return).
        return string_at(compressed_values.data, compressed_values.len)
    finally:
        # This block ensures we free the Zig-allocated memory.
        if compressed_values.data:
            __library.freeCompressedValues(byref(compressed_values))


def decompress(values: bytes) -> List[float]:
    """Decompress a TerseTS-compressed buffer into a list of floats.

    This function restores the original float64 sequence from a compressed
    TerseTS payload. When NumPy is installed, it uses a zero-copy view of
    the input buffer and an efficient conversion to Python list.

    Args:
      values: Compressed buffer as `bytes`, `bytearray`, or `memoryview`.
        Passing `bytes` yields the best performance.

    Returns:
      Decompressed data as a `list[float]`.

    Raises:
      TypeError: If `values` is not a `bytes`, `bytearray`, or `memoryview`.
      RuntimeError: If the native `decompress` call fails (non-zero error code).

    Notes:
      - **Zero-copy path**: when NumPy is installed, `np.frombuffer` maps
        the compressed payload directly without copying.
      - **Copy path**: without NumPy, a temporary contiguous `c_ubyte[]`
        buffer is built before calling the native decompressor.

    Examples:
      >>> blob = compress([1.0, 2.0, 3.0], Method.SwingFilter, 0.01)
      >>> decompress(blob)
      [1.0, 2.0, 3.0]
    """
    # Validate input type.
    if not isinstance(values, (bytes, bytearray, memoryview)):
        raise TypeError(
            "decompress(values): values must be bytes, bytearray, or memoryview"
        )

    # Prepare a __CompressedValues view and keep a Python reference alive during the call.
    # The reference to the underlying buffer (NumPy view or ctypes array) must stay alive
    # so that the garbage collector does not release it while the native code reads from it.
    compressed_values = __CompressedValues()

    if _INSTALLED_NUMPY:
        # Zero-copy view of the compressed input; avoids building a c_ubyte[] buffer.
        view = np.frombuffer(values, dtype=np.uint8)
        compressed_values.data = view.ctypes.data_as(POINTER(c_ubyte))
        compressed_values.len = view.size
    else:
        # No NumPy: make a temporary c_ubyte[] copy to pass into the native layer.
        buf = (c_ubyte * len(values)).from_buffer_copy(values)
        compressed_values.data = cast(buf, POINTER(c_ubyte))
        compressed_values.len = len(values)

    # Prepare destination struct for decompressed values.
    uncompressed_values = __UncompressedValues()
    try:
        # Call native library.
        tersets_error = __library.decompress(
            compressed_values, byref(uncompressed_values)
        )
        if tersets_error != 0:
            raise RuntimeError(f"decompress failed: {tersets_error}")

        if _INSTALLED_NUMPY:
            # Create a NumPy view over the returned f64s.
            # IMPORTANT ON OWNERSHIP/LIFETIME:
            # - `as_array(ptr, shape=...)` makes a *view* onto Zig-owned memory; it does not copy.
            # - Returning this array would expose a dangling pointer once Zig frees/repurposes the buffer.
            # - We therefore materialize a Python-owned object via `.tolist()`, which *copies* the data
            #   and fully detaches from the Zig buffer. This keeps the API safe and GC-friendly.
            return np.ctypeslib.as_array(
                uncompressed_values.data, shape=(uncompressed_values.len,)
            ).tolist()

        # ctypes-only: slice the POINTER range into a Python list.
        # `uncompressed_values.data[:len]` copies the values into a new Python list,
        # so the result is owned by Python and independent from the Zig buffer.
        return uncompressed_values.data[: uncompressed_values.len]
    finally:
        # This block ensures we free the Zig-allocated memory.
        __library.freeUncompressedValues(byref(uncompressed_values))
