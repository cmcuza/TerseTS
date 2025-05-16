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
from typing import Union
from enum import Enum, IntEnum, unique
from ctypes import (
    cdll,
    Structure,
    c_byte,
    c_uint8,
    c_float,
    c_double,
    c_size_t,
    c_void_p,
    POINTER,
    byref,
    cast,
    c_int32,
)

# -----------------------------------------------------------------------------
# Internal: Load the shared library
# -----------------------------------------------------------------------------

def __load_library():
    if sys.platform == "win32":
        library_name = "tersets.pyd"
    else:
        library_name = "tersets" + sysconfig.get_config_var("SHLIB_SUFFIX")

    library_folder = pathlib.Path(__file__).parent.parent.resolve()
    library_path = library_folder / library_name
    if library_path.exists():
        return cdll.LoadLibrary(str(library_path))

    # Fallback: use Zig build location.
    repository_root = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
    if sys.platform == "win32":
        library_path = repository_root / "zig-out" / "bin" / "tersets.dll"
    elif sys.platform == "darwin":
        library_path = repository_root / "zig-out" / "lib" / "tersets.dylib"
    else:
        library_path = repository_root / "zig-out" / "lib" / library_name

    return cdll.LoadLibrary(str(library_path))

__library = __load_library()

# -----------------------------------------------------------------------------
# C Structs & Enums
# -----------------------------------------------------------------------------

class ErrorBoundType(IntEnum):
    ABS = 0
    RELATIVE = 1

class CostFunction(IntEnum):
    RMSE = 0
    LINF = 1

class _UncompressedValues(Structure):
    _fields_ = [("data", POINTER(c_double)), ("len", c_size_t)]

class _CompressedValues(Structure):
    _fields_ = [("data", POINTER(c_byte)), ("len", c_size_t)]

class BasicParams(Structure):
    _fields_ = [
        ("error_bound", c_float),
    ]

class FunctionalParams(Structure):
    _fields_ = [
        ("error_bound_type", c_uint8),  # ErrorBoundType
        ("error_bound", c_float),
    ]

class HistogramParams(Structure):
    _fields_ = [
        ("maximum_buckets", c_size_t),
    ]

class LineSimplificationParams(Structure):
    _fields_ = [
        ("cost_function", c_uint8),  # CostFunction
        ("error_bound", c_float),
    ]

class _Configuration(Structure):
    _fields_ = [
        ("method", c_uint8),
        ("parameters", c_void_p),
    ]

@unique
class Method(IntEnum):
    PoorMansCompressionMidrange = 0
    PoorMansCompressionMean = 1
    SwingFilter = 2
    SwingFilterDisconnected = 3
    SlideFilter = 4
    SimPiece = 5
    PiecewiseConstantHistogram = 6
    PiecewiseLinearHistogram = 7
    VisvalingamWhyatt = 8
    IdentityCompression = 9

@unique
class TerseTSError(IntEnum):
    UnknownMethod = 1
    UnsupportedInput = 2
    UnsupportedErrorBound = 3
    UnsupportedParameters = 4
    ItemNotFound = 5
    OutOfMemory = 6
    EmptyConvexHull = 7
    EmptyQueue = 8

_error_messages = {
    1: "Unknown compression method.",
    2: "Unsupported or empty input.",
    3: "Unsupported or negative error bound.",
    4: "Unsupported or missing configuration parameters.",
    5: "Item not found during decompression.",
    6: "Out of memory.",
    7: "Convex hull was empty.",
    8: "Internal queue was unexpectedly empty.",
}

# -----------------------------------------------------------------------------
# Function Signatures
# -----------------------------------------------------------------------------

__library.compress.argtypes = [_UncompressedValues, POINTER(_CompressedValues), _Configuration]
__library.compress.restype = c_int32

__library.decompress.argtypes = [_CompressedValues, POINTER(_UncompressedValues)]
__library.decompress.restype = c_int32

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def compress(values: List[float], method: Method, params: Union[float, BasicParams, FunctionalParams, HistogramParams, LineSimplificationParams]) -> bytes:
    """Compress a list of floats using the specified method and parameter configuration.

    You may pass:
    - a float error_bound (used as BasicParams)
    - an instance of a full parameter struct (e.g., FunctionalParams)
    """
    if not isinstance(method, Method):
        raise TypeError(f"Invalid method: {method}")

    # Convert float to BasicParams if needed
    if isinstance(params, float):
        params_struct = BasicParams(error_bound=params)
    elif isinstance(params, (BasicParams, FunctionalParams, HistogramParams, LineSimplificationParams)):
        params_struct = params
        is_default = 0
    else:
        raise TypeError(f"Invalid params type: {type(params)}")

    # Set up uncompressed values
    array_type = c_double * len(values)
    uncompressed_values = _UncompressedValues()
    uncompressed_values.data = array_type(*values)
    uncompressed_values.len = len(values)

    compressed_values = _CompressedValues()
    config = _Configuration(
        method=method.value,
        is_default=is_default,
        parameters=cast(c_void_p, byref(params_struct))
    )

    err = __library.compress(uncompressed_values, byref(compressed_values), config)
    if err != 0:
        raise RuntimeError(_error_messages.get(err, f"Unknown error code: {err}"))

    return bytes(compressed_values.data[:compressed_values.len])


def decompress(values: bytes) -> List[float]:
    """Decompress a byte sequence into a list of floats."""
    compressed_values = _CompressedValues()
    compressed_values.data = (c_byte * len(values))(*values)
    compressed_values.len = len(values)

    decompressed_values = _UncompressedValues()
    err = __library.decompress(compressed_values, byref(decompressed_values))
    if err != 0:
        raise RuntimeError(_error_messages.get(err, f"Unknown error code: {err}"))

    return list(decompressed_values.data[:decompressed_values.len])
