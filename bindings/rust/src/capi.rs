// Copyright 2026 TerseTS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The C API provided by the TerseTS library.

use std::os::raw::c_char;

// TODO: implement Drop that calls freeUncompressedValues().
#[repr(C)]
pub(super) struct UncompressedValues {
    pub(super) data: *const f64,
    pub(super) len: usize,
}

// TODO: implement Drop that calls freeCompressedValues().
#[repr(C)]
pub(super) struct CompressedValues {
    pub(super) data: *mut u8,
    pub(super) len: usize,
}

unsafe extern "C" {
    pub fn compress(
        uncompressed_values: UncompressedValues,
        compressed_values: *mut CompressedValues,
        method: u8,
        configuration: *const c_char,
    ) -> i32;

    pub fn decompress(
        compressed_values: CompressedValues,
        uncompressed_values: *mut UncompressedValues,
    ) -> i32;

    pub fn freeCompressedValues(compressed_values: *mut CompressedValues);

    pub fn freeUncompressedValues(uncompressed_values: *mut UncompressedValues);
}
