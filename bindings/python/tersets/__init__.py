""" Python bindings for the TerseTS library. """

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


import os
import platform
from typing import List
from ctypes import cdll, Structure, c_byte, c_double, c_size_t, POINTER, pointer


# Private Functions.
def __load_library():
    """ Locates the correct library for this system and loads it. """

    # Compute the path to the current working directory to locate the library.
    script_folder = os.path.dirname(os.path.abspath(__file__))

    # Determine the architecture and operating system of the system.
    cpu_architecture = platform.machine()
    operating_system = platform.system()

    # Load the correct shared library for the operating system.
    if cpu_architecture == "x86_64" and operating_system == "Linux":
        library_path = script_folder + "/../../../zig-out/lib/libtersets.so.0.0.1"
    elif cpu_architecture == "arm64" and operating_system == "Darwin":
        library_path = script_folder + "/../../../zig-out/lib/libtersets.0.0.1.dylib"
    elif cpu_architecture == "x86_64" and operating_system == "Darwin":
        library_path = script_folder + "/../../../zig-out/lib/libtersets.0.0.1.dylib"
    elif cpu_architecture == "AMD64" and operating_system == "Windows":
        library_path = script_folder + "\\..\\..\\..\\zig-out\\lib\\tersets.dll"
    elif cpu_architecture == "x86_64" and operating_system == "Windows":
        library_path = script_folder + "\\..\\..\\..\\zig-out\\lib\\tersets.dll"
    else:
        raise ValueError(
            f"{operating_system} on {cpu_architecture} is not supported")

    return cdll.LoadLibrary(library_path)


# A global variable is used for the library so it is only initialized once and
# can be easily used by the public functions without users having to pass it.
__library = __load_library()


# Private Types.
class __InputData(Structure):
    _fields_ = [("values", POINTER(c_double)), ("len", c_size_t)]

class __OutputData(Structure):
    _fields_ = [("values", POINTER(c_byte)), ("len", c_size_t)]


# Public Functions.
def compress(values: List[float]) -> bytes:
    """ Compresses values. """

    input_data = __InputData()
    input_data.values = (c_double * len(values))(*values)
    input_data.len = len(values)
    input_data_ptr = pointer(input_data)

    output_data = __OutputData()
    output_data_ptr = pointer(output_data)

    error = __library.compress(input_data_ptr, output_data_ptr)

    return error
