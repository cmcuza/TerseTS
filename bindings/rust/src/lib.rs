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

//! Rust bindings for the TerseTS library.

mod capi;
mod error;

use std::ffi::CString;
use std::ptr;
use std::slice;

use capi::{CompressedValues, UncompressedValues};
pub use error::{Result, TerseTSError};

/// Mirror TerseTS Method Enum.
#[repr(u8)]
pub enum Method {
    PoorMansCompressionMidrange,
    PoorMansCompressionMean,
    SwingFilter,
    SwingFilterDisconnected,
    SlideFilter,
    SimPiece,
    PiecewiseConstantHistogram,
    PiecewiseLinearHistogram,
    ABCLinearApproximation,
    VisvalingamWhyatt,
    SlidingWindow,
    BottomUp,
    MixPiece,
    BitPackedQuantization,
    RunLengthEncoding,
    NonLinearApproximation,
    SerfQT,
    DiscreteFourierTransform,
}

/// Compress a slice of [`f64`] in `uncompressed_values` to a [`Vec`] of [`u8`] with a TerseTS
/// compression `method` according to `configuration`. If an error occurs it is returned.
pub fn compress(
    uncompressed_values: &[f64],
    method: Method,
    configuration: &str,
) -> Result<Vec<u8>> {
    let uncompressed_values_struct = UncompressedValues {
        data: uncompressed_values.as_ptr(),
        len: uncompressed_values.len(),
    };

    let mut compressed_values_struct = CompressedValues {
        data: ptr::null_mut(),
        len: 0,
    };

    let configuration = CString::new(configuration)?;

    let tersets_error = unsafe {
        capi::compress(
            uncompressed_values_struct,
            &mut compressed_values_struct,
            method as u8,
            configuration.as_ptr(),
        )
    };

    if tersets_error != 0 {
        unsafe { capi::freeCompressedValues(&mut compressed_values_struct) };
        return Err(TerseTSError::TerseTS(tersets_error));
    }

    let compressed_values = unsafe {
        slice::from_raw_parts(
            compressed_values_struct.data as *const u8,
            compressed_values_struct.len,
        )
        .to_vec()
    };

    unsafe { capi::freeCompressedValues(&mut compressed_values_struct) };

    Ok(compressed_values)
}

/// Decompress a TerseTS-compressed slice of [`u8`] to a [`Vec`] of [`f64`]. If an error occurs it
/// is returned.
pub fn decompress(compressed_values: &[u8]) -> Result<Vec<f64>> {
    let compressed_values_struct = CompressedValues {
        data: compressed_values.as_ptr() as *mut u8,
        len: compressed_values.len(),
    };

    let mut uncompressed_values_struct = UncompressedValues {
        data: ptr::null(),
        len: 0,
    };

    let tersets_error =
        unsafe { capi::decompress(compressed_values_struct, &mut uncompressed_values_struct) };

    if tersets_error != 0 {
        unsafe { capi::freeUncompressedValues(&mut uncompressed_values_struct) };
        return Err(TerseTSError::TerseTS(tersets_error));
    }

    let uncompressed_values = unsafe {
        slice::from_raw_parts(
            uncompressed_values_struct.data,
            uncompressed_values_struct.len,
        )
        .to_vec()
    };

    unsafe { capi::freeUncompressedValues(&mut uncompressed_values_struct) };

    Ok(uncompressed_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_loads_tersets_without_error_bound() {
        let compressed_values = compress(
            &[10.0, 20.0, 30.0, 40.0, 50.0],
            Method::SwingFilter,
            r#"{ "abs_error_bound": 0.0 }"#,
        )
        .unwrap();

        let uncompressed_values = decompress(&compressed_values).unwrap();
        assert_eq!(uncompressed_values, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    }
}
