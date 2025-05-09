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
from ctypes import cdll, Structure, c_byte, c_float, c_double, c_size_t, POINTER, byref


# Private Functions.
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

    library_path = next(library_folder.glob("*" + library_name))
    if library_path.exists():
        return cdll.LoadLibrary(str(library_path))


# A global variable is used for the library so it is only initialized once and
# can be easily used by the public functions without users having to pass it.
__library = __load_library()


# Private Types.
class __UncompressedValues(Structure):
    _fields_ = [("data", POINTER(c_double)), ("len", c_size_t)]


class __CompressedValues(Structure):
    _fields_ = [("data", POINTER(c_byte)), ("len", c_size_t)]


class __Configuration(Structure):
    _fields_ = [("method", c_byte), ("error_bound", c_float)]


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
    VisvalingamWhyatt = 8

# Public Functions.
def compress(values: List[float], method: Method, error_bound: float) -> bytes:
    """Compresses values."""

    uncompressed_values = __UncompressedValues()
    uncompressed_values.data = (c_double * len(values))(*values)
    uncompressed_values.len = len(values)

    compressed_values = __CompressedValues()

    if type(method) != Method:
        # Method does not exists, raise error, and show available options.
        available_methods = ", ".join([member.name for member in Method])
        raise TypeError(
            f"'{method}' is not a valid TerseTS Method. Available method names are: {available_methods}"
        )

    configuration = __Configuration(method.value, error_bound)

    error = __library.compress(
        uncompressed_values, byref(compressed_values), configuration
    )

    if error == 1:
        raise ValueError("Unknown error.")

    return compressed_values.data[: compressed_values.len]


def decompress(values: bytes) -> List[float]:
    """Decompresses values."""

    compressed_values = __CompressedValues()
    compressed_values.data = (c_byte * len(values))(*values)
    compressed_values.len = len(values)

    decompressed_values = __UncompressedValues()

    error = __library.decompress(compressed_values, byref(decompressed_values))

    if error == 1:
        raise ValueError("Unknown decompression method.")

    return decompressed_values.data[: decompressed_values.len]
