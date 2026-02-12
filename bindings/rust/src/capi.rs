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

/// A pointer to uncompressed values and the number of values.
#[repr(C)]
pub(super) struct UncompressedValues {
    pub(super) data: *const f64,
    pub(super) len: usize,
}

impl Drop for UncompressedValues {
    fn drop(&mut self) {
        unsafe { freeUncompressedValues(self) };
    }
}

/// A pointer to compressed values and the number of bytes.
#[repr(C)]
pub(super) struct CompressedValues {
    pub(super) data: *mut u8,
    pub(super) len: usize,
}

impl Drop for CompressedValues {
    fn drop(&mut self) {
        unsafe { freeCompressedValues(self) };
    }
}

unsafe extern "C" {
    /// Compress an `UncompressedValues` with a TerseTS compression `method` according to `configuration`.
    pub fn compress(
        uncompressed_values: UncompressedValues,
        compressed_values: *mut CompressedValues,
        method: u8,
        configuration: *const c_char,
    ) -> i32;

    /// Decompress a TerseTS-compressed `CompressedValues` into an `UncompressedValues`.
    pub fn decompress(
        compressed_values: CompressedValues,
        uncompressed_values: *mut UncompressedValues,
    ) -> i32;

    /// Free a `CompressedValues`.
    pub fn freeCompressedValues(compressed_values: *mut CompressedValues);

    /// Free an `UncompressedValues`.
    pub fn freeUncompressedValues(uncompressed_values: *mut UncompressedValues);
}
