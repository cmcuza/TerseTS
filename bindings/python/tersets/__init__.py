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

from __future__ import annotations
import sys
import pathlib
import sysconfig
import json
from typing import List, Tuple, Union, Dict, Any
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
    import numpy

    _INSTALLED_NUMPY = True
except Exception:
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


class __CoefficientsValues(Structure):
    _fields_ = [("data", POINTER(c_double)), ("len", c_size_t)]


class __IndicesValues(Structure):
    _fields_ = [("data", POINTER(c_size_t)), ("len", c_size_t)]


# Declare function signatures to avoid silent arguments mismatch.
__library.compress.argtypes = [
    __UncompressedValues,
    POINTER(__CompressedValues), 
    c_uint8, 
    c_char_p,
]
__library.compress.restype = c_int
__library.decompress.argtypes = [__CompressedValues, POINTER(__UncompressedValues)]
__library.decompress.restype = c_int

__library.extract.argtypes = [
    __CompressedValues, 
    POINTER(__IndicesValues), 
    POINTER(__CoefficientsValues),
]
__library.extract.restype  = c_int

__library.rebuild.argtypes = [
    __IndicesValues, 
    __CoefficientsValues, 
    POINTER(__CompressedValues), 
    c_ubyte,
]
__library.rebuild.restype  = c_int


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
    SerfQT = 16
    BitPackedBUFF = 17


# Public API. 
def compress(
    values: Union["numpy.ndarray", List[float], Tuple[float, ...]],
    method: Method,
    configuration: Union[Dict[str, Any], str],
) -> Union[bytes, "numpy.ndarray"]:
    """Compress a sequence of float64 values with a selected TerseTS method.

    This function uses a zero-copy fast path when `values` is a NumPy `ndarray`
    of dtype `float64` and C-contiguous. Otherwise, it falls back to a ctypes
    copy for Python lists/tuples of floats.

    Args:
      values: Either a `numpy.ndarray` of dtype `float64` (C-contiguous) or a
        `list`/`tuple` of floats.
      method: A `Method` enum value selecting the compressor.
      configuration: Either a `dict` (will be JSON-encoded) or a JSON `str`.

    Returns:
      If NumPy is installed: `numpy.ndarray[uint8]` containing the compressed representation.
      Otherwise: `bytes` containing the compressed representation. The returned array contains
      a trailing byte storing the method which is used during decompression.

    Raises:
      TypeError: If `method` is not a `Method` or `values`/`configuration` have
        unsupported types.
      RuntimeError: If the native `compress` call returns a non-zero error code.

    Examples:
      >>> configuration = {"abs_error_bound": 0.1}
      >>> compress([1.0, 2.08, 2.96], Method.SwingFilter, configuration)
      array([ ... ], dtype=uint8)   # if NumPy installed
      b'\x00\xf0 ... \x00\x02'      # without NumPy
    """
    # Validate compressor method type.
    if not isinstance(method, Method):
        available = ", ".join(m.name for m in Method)
        raise TypeError(
            f"{method!r} is not a valid TerseTS Method. Available: {available}"
        )

    # Prepare an __UncompressedValues view and keep a Python reference alive during the call.
    # The reference to the source array (NumPy array or ctypes array) must stay alive so the
    # garbage collector does not free the underlying memory while the native code reads from it.
    uncompressed_values = __UncompressedValues()

    if _INSTALLED_NUMPY and isinstance(values, numpy.ndarray):
        # Ensure dtype and memory layout are correct for zero-copy access.
        # If `values` is already float64 and C-contiguous, no copy is made.
        # Otherwise, numpy.ascontiguousarray() converts it (potentially copying)
        # so we can safely pass its data pointer directly to the native layer.
        if values.dtype != numpy.float64 or not values.flags["C_CONTIGUOUS"]:
            values = numpy.ascontiguousarray(values, dtype=numpy.float64)
        uncompressed_values.data = values.ctypes.data_as(POINTER(c_double))
        uncompressed_values.len = values.size
    elif isinstance(values, (list, tuple)):
        # Build a contiguous C array of f64 values from the Python sequence.
        # Python lists/tuples store references to float objects, not the raw
        # float64 values themselves, so the data is not contiguous in memory.
        # We copy each element into a new C double[] array to ensure the
        # native layer can read a continuous block of numeric values.
        buffer = (c_double * len(values))(*values)
        uncompressed_values.data = buffer
        uncompressed_values.len = len(values)
    else:
        # Invalid input type.
        raise TypeError(
            "values must be a numpy.ndarray[float64] that is C-contiguous, or a list/tuple of floats."
        )

    # Prepare compressed values array.
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
        
        if _INSTALLED_NUMPY:
            # The native layer allocates `compressed_values.data`. 
            # Copy into NumPy-owned memory before freeing the Zig arrays.
            return numpy.ctypeslib.as_array(
                cast(compressed_values.data, POINTER(c_ubyte)),
                shape=(compressed_values.len,),
            ).copy()

        # No NumPy. Copy the compressed bytes into Python-owned memory (safe to return).
        return string_at(compressed_values.data, compressed_values.len)
    finally:
        # This block ensures we free the Zig-allocated memory.
        if compressed_values.data:
            __library.freeCompressedValues(byref(compressed_values))


def decompress(
    values: Union[bytes, bytearray, memoryview, "numpy.ndarray"],
) -> Union[List[float], "numpy.ndarray"]:
    """Decompress a TerseTS-compressed array into a list of floats.

    This function restores the float64 values from a compressed TerseTS compressed 
    representation. When NumPy is installed, it uses a zero-copy view of the input array and
    an efficient conversion to Python list. The function accepts compressed input as `bytes`,
    `bytearray`, `memoryview`, or a `numpy.ndarray[uint8]`. If NumPy is installed, returns a
    `numpy.ndarray[float64]`, which is typically faster for downstream numerical processing
    than a Python list. Without NumPy, it returns a Python `list[float]`.

    Args:
      values: Compressed array as `bytes`, `bytearray`, `memoryview`, or 
        a `numpy.ndarray` of dtype `uint8` (C-contiguous). 

    Returns:
      If NumPy is installed: Decompressed data as `numpy.ndarray[float64]`.
      Otherwise: Decompressed data as `list[float]`.

    Raises:
      TypeError: If `values` is not a supported array type.
      RuntimeError: If the native `decompress` call fails (non-zero error code).

    Examples:
      >>> configuration = {"abs_error_bound": 0.1}
      >>> blob = compress([1.0, 2.08, 2.96], Method.SwingFilter, configuration)
      >>> decompress(blob)
      array([1.0, 2.0, 3.0]) # if NumPy installed
      [1.0, 2.0, 3.0]        # without NumPy
    """
    # Validate input type.
    if _INSTALLED_NUMPY and isinstance(values, numpy.ndarray):
        # Explicit NumPy support. Validate dtype and layout.
        if values.dtype != numpy.uint8:
            raise TypeError("decompress(values): NumPy array must have dtype=np.uint8")
        if values.ndim != 1:
            raise TypeError("decompress(values): NumPy array must be 1-dimensional")
        if not values.flags["C_CONTIGUOUS"]:
            raise TypeError("decompress(values): NumPy array must be C-contiguous")
    elif not isinstance(values, (bytes, bytearray, memoryview)):
        raise TypeError(
            "decompress(values): values must be bytes, bytearray, memoryview or a NumPy uint8 array"
        )

    # Prepare a __CompressedValues view and keep a Python reference alive during the call.
    # The reference to the underlying array (NumPy view or ctypes array) must stay alive
    # so that the garbage collector does not release it while the native code reads from it.
    compressed_values = __CompressedValues()

    if _INSTALLED_NUMPY:
        # Zero-copy view of the compressed input which avoids building a c_ubyte[] array.
        view = numpy.frombuffer(values, dtype=numpy.uint8)
        compressed_values.data = view.ctypes.data_as(POINTER(c_ubyte))
        compressed_values.len = view.size
    else:
        # No NumPy: make a temporary c_ubyte[] copy to pass into the native layer.
        values_buffer = (c_ubyte * len(values)).from_buffer_copy(values)
        compressed_values.data = cast(values_buffer, POINTER(c_ubyte))
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
            # - `as_array(ptr, shape=...)` makes a *view* onto Zig-owned memory.
            # - Copy before freeing the Zig array in `finally`.
            return numpy.ctypeslib.as_array(
                uncompressed_values.data, shape=(uncompressed_values.len,)
            ).copy()

        # `uncompressed_values.data[:len]` copies the values into a new Python list,
        # so the result is owned by Python and independent from the Zig array.
        return uncompressed_values.data[: uncompressed_values.len]
    finally:
        # This block ensures we free the Zig-allocated memory.
        __library.freeUncompressedValues(byref(uncompressed_values))


def extract(
    values: Union[bytes, bytearray, memoryview, "numpy.ndarray"],
) -> Union[
    Tuple[List[int], List[float]],
    Tuple["numpy.ndarray", "numpy.ndarray"],
]:
    """Extract indices and coefficients from a compressed TerseTS compressed representation.

    This function parses a compressed representation and returns its internal
    representation as two arrays: indices and coefficients. In this context, 
    indices are method-dependent metadata need to reconstruct the original time series,
    e.g., indices of key points, segment boundaries, or histogram bin indices.
    Coefficients are the floating-point values associated with these indices.
    Pass `values` as `bytes` for the fastest processing using Python or install 
    NumPy to enable efficient memory handling and avoid unnecessary copies.

    Args:
      values: Compressed array as `bytes`, `bytearray`, `memoryview`, or (if NumPy
        is installed) a `numpy.ndarray` of dtype `uint8`.

    Returns:
      If NumPy is installed:
        Tuple of (`indices: numpy.ndarray[uintp]`, `coefficients: numpy.ndarray[float64]`).
      Otherwise:
        Tuple of (`indices: list[int]`, `coefficients: list[float]`).

    Raises:
      TypeError: If `values` is not a supported array type.
      RuntimeError: If the native `extract` call fails (non-zero error code).

    Notes:
      - Output layouts depend on the compression method used.
      - See TerseTS's documentation for details on indices/coefficients layouts.
      - Output arrays are copied into Python/NumPy-owned memory before the Zig
        arrays are freed.

    Examples:
      >>> configuration = {"abs_error_bound": 0.1}
      >>> blob = compress([1.0, 2.08, 2.96], Method.SwingFilter, configuration)
      >>> extract(blob)
      (array([3], dtype=uintp), array([1., 3.]))   # if NumPy installed
      ([3], [1.0, 3.0])                            # without NumPy
    """
    compressed_values = __CompressedValues()

    if _INSTALLED_NUMPY and isinstance(values, numpy.ndarray):
        # Explicit NumPy support. Validate dtype and layout.
        if values.dtype != numpy.uint8:
            raise TypeError("NumPy array must have dtype=np.uint8")
        if values.ndim != 1:
            raise TypeError("NumPy array must be 1-dimensional")
        if not values.flags["C_CONTIGUOUS"]:
            raise TypeError("NumPy array must be C-contiguous")

        compressed_values.data = values.ctypes.data_as(POINTER(c_ubyte))
        compressed_values.len = values.size

    elif isinstance(values, (bytes, bytearray, memoryview)):
        if _INSTALLED_NUMPY:
            # Zero-copy via NumPy view.
            view = numpy.frombuffer(values, dtype=numpy.uint8)
            compressed_values.data = view.ctypes.data_as(POINTER(c_ubyte))
            compressed_values.len = view.size
        else:
            # ctypes fallback.
            values_buffer = (c_ubyte * len(values)).from_buffer_copy(values)
            compressed_values.data = cast(values_buffer, POINTER(c_ubyte))
            compressed_values.len = len(values)

    else:
        raise TypeError(
            "Values must be bytes, bytearray, memoryview, or a NumPy uint8 array"
        )

    indices_values = __IndicesValues()
    coefficients_values = __CoefficientsValues()

    try:
        err = __library.extract(
            compressed_values, 
            byref(indices_values), 
            byref(coefficients_values),
        )
        if err != 0:
            raise RuntimeError(f"extract failed: {err}")

        if _INSTALLED_NUMPY:
            # Create views onto native memory, then copy into NumPy-owned arrays
            # before we free the native allocations in `finally`.
            indices = numpy.ctypeslib.as_array(indices_values.data, 
                                                  shape=(indices_values.len,)).copy()
            coefficients = numpy.ctypeslib.as_array(coefficients_values.data, 
                                                    shape=(coefficients_values.len,)).copy()
            return indices, coefficients

        # No NumPy: copy into Python lists.
        return (
            list(indices_values.data[: indices_values.len]),
            list(coefficients_values.data[: coefficients_values.len]),
        )
    finally:
        if indices_values.data:
            __library.freeIndicesValues(byref(indices_values))
        if coefficients_values.data:
            __library.freeCoefficientValues(byref(coefficients_values))
        

def rebuild(
    indices: Union["numpy.ndarray", List[int], Tuple[int, ...]],
    coefficients: Union["numpy.ndarray", List[float], Tuple[float, ...]],
    method: Method,
) -> Union[bytes, "numpy.ndarray"]:
    """Rebuild a compressed TerseTS representation from extracted indices and coefficients.

    This function is the inverse of :func:`extract` and constructs a valid
    binary compressed representation from the method-specific arrays `indices` 
    and `coefficients`. In this context, indices are method-dependent metadata
    needed to reconstruct the original time series, e.g., indices of key points, 
    segment boundaries, or histogram bin indices. Coefficients are the floating-point
    values associated with these indices. The returned representation is identical in
    format to what :func:`compress` would produce for the same method. 
    
    The meaning and required structure of `indices` and `coefficients`
    depend on the selected compression method. 
    
    TerseTS performs structural validation and will return an error 
    if the arrays do not satisfy the invariants of the target `method`. 
    
    When NumPy is installed, the function uses a zero-copy view of both input arrays
    (if they are C-contiguous and of the correct dtype), making this significantly
    faster for large arrays. Otherwise, it falls back to building temporary ctypes arrays.

    Args:
      indices: Method-dependent integer metadata. May be a NumPy array of
        dtype `numpy.uintp` (C-contiguous) or a Python `list` / `tuple` of
        Python integers.
      coefficients: Method-dependent floating-point coefficients. May be a
        NumPy array of dtype `float64` (C-contiguous) or a Python `list`/`tuple` of floats.
      method: A `Method` enum value selecting the target compressor.

    Returns:
      If NumPy is installed: `numpy.ndarray[uint8]` containing the rebuilt representation.
      Otherwise: `bytes` containing the rebuilt representation.

    Raises:
      TypeError:
        * If `method` is not a `Method` enum value.
        * If `indices` is not an ndarray/list/tuple of valid integers.
        * If `coefficients` is not an ndarray/list/tuple of floats.
        * If NumPy arrays do not match required dtypes/layouts.
      RuntimeError:
        If the native `rebuild` call fails (structural validation failed).

    Notes:
      - NumPy-based zero-copy input requires arrays to be C-contiguous and of the correct dtype.
      - Output is copied into Python/NumPy-owned memory before freeing the Zig array.
    
    Examples:
      >>> configuration = {"abs_error_bound": 0.1}
      >>> blob = compress([1.0, 2.08, 2.96], Method.SwingFilter, configuration)
      >>> indices, coefficients = extract(blob)
      >>> rebuilt = rebuild(indices, coefficients, Method.SwingFilter)
      >>> decompress(rebuilt)
      array([1., 2., 3.])   # if NumPy installed
    """
    if not isinstance(method, Method):
        available = ", ".join(m.name for m in Method)
        raise TypeError(
            f"{method!r} is not a valid TerseTS Method. Available: {available}"
        )

    compressed_values = __CompressedValues()

    # Prepare indices (size_t*).
    indices_values = __IndicesValues()
    if _INSTALLED_NUMPY and isinstance(indices, numpy.ndarray):
        if indices.dtype != numpy.uintp:
            raise TypeError("rebuild(): 'indices' NumPy array must have dtype=np.uintp")
        if indices.ndim != 1:
            raise TypeError("rebuild(): 'indices' NumPy array must be 1-dimensional")
        if not indices.flags["C_CONTIGUOUS"]:
            raise TypeError("rebuild(): 'indices' NumPy array must be C-contiguous")

        indices_values.data = indices.ctypes.data_as(POINTER(c_size_t))
        indices_values.len = indices.size
    elif isinstance(indices, (list, tuple)):
        indices_buffer = (c_size_t * len(indices))(*indices)
        indices_values.data = indices_buffer
        indices_values.len = len(indices)
    else:
        raise TypeError("rebuild(): 'indices' must be ndarray[uintp] or list/tuple[int]")

    # Prepare coefficients (double*).
    coefficients_values = __CoefficientsValues()
    if _INSTALLED_NUMPY and isinstance(coefficients, numpy.ndarray):
        if coefficients.dtype != numpy.float64:
            raise TypeError("rebuild(): 'coefficients' NumPy array must have dtype=np.float64")
        if coefficients.ndim != 1:
            raise TypeError("rebuild(): 'coefficients' NumPy array must be 1-dimensional")
        if not coefficients.flags["C_CONTIGUOUS"]:
            raise TypeError("rebuild(): 'coefficients' NumPy array must be C-contiguous")

        coefficients_values.data = coefficients.ctypes.data_as(POINTER(c_double))
        coefficients_values.len = coefficients.size
    elif isinstance(coefficients, (list, tuple)):
        coefficients_buffer = (c_double * len(coefficients))(*coefficients)
        coefficients_values.data = coefficients_buffer
        coefficients_values.len = len(coefficients)
    else:
        raise TypeError("rebuild(): 'coefficients' must be ndarray[float64] or list/tuple[float]")

    try:
        err = __library.rebuild(
            indices_values,
            coefficients_values,
            byref(compressed_values),
            method.value,
        )
        if err != 0:
            raise RuntimeError(f"rebuild failed: {err}")

        if _INSTALLED_NUMPY:
            # Copy native output into NumPy-owned memory before freeing Zig array.
            out_view = numpy.ctypeslib.as_array(
                cast(compressed_values.data, POINTER(c_ubyte)),
                shape=(compressed_values.len,),
            )
            return out_view.copy()

        return string_at(compressed_values.data, compressed_values.len)

    finally:
        if compressed_values.data:
            __library.freeCompressedValues(byref(compressed_values))