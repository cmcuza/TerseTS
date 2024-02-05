""" Python bindings for the TerseTS Zig library. """

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
from sys import platform
from ctypes import cdll, Structure, c_byte, c_double, c_size_t, POINTER, pointer


# Compute the path to the current working directory to locate the Zig library.
script_folder = os.path.dirname(os.path.abspath(__file__))


# Load the correct shared library for the operating system, assumes .
if platform == "linux" or platform == "linux2":
    library_path = script_folder + "/../zig-out/lib/libtersets.so.0.0.1"
elif platform == "darwin":
    library_path = script_folder + "/../zig-out/lib/libtersets.0.0.1.dylib"
elif platform == "win32":
    library_path = script_folder + "\\..\\zig-out\\lib\\tersets.dll"
else:
    raise ValueError(f"Linux, macOS, and Windows are supported, not {platform}")

tersets = cdll.LoadLibrary(library_path)


# Compress by passing the struct containing a pointer to floating-point values.
class InputData(Structure):
    _fields_ = [("values", POINTER(c_double)), ("len", c_size_t)]

class OutputData(Structure):
    _fields_ = [("values", POINTER(c_byte)), ("len", c_size_t)]

py_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
c_array = (c_double * len(py_list))(*py_list)

input_data = InputData()
input_data.values = c_array
input_data.len = len(py_list)
input_data_ptr = pointer(input_data)

output_data = OutputData()
output_data_ptr = pointer(output_data)

tersets.compress(input_data_ptr, output_data_ptr)
