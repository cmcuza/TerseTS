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

use capi::{CompressedValues, UncompressedValues};

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
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
}

#[derive(Debug)]
pub struct TerseError {
    pub code: i32,
}

type Result<T> = std::result::Result<T, TerseError>;


pub fn compress(uncompressed: &[f64], method: Method, configuration: &str) -> Result<Vec<u8>> {
    let uncompressed_values = UncompressedValues { data: uncompressed.as_ptr(), len: uncompressed.len() };

    let mut compressed_values = CompressedValues { data: std::ptr::null_mut(), len: 0 };

    let configuration = std::ffi::CString::new(configuration).map_err(|_| TerseError { code: -1001 })?;

    let code = unsafe {
        capi::compress(uncompressed_values, &mut compressed_values, method as u8, configuration.as_ptr())
    };

    if code != 0 {
        unsafe { capi::freeCompressedValues(&mut compressed_values) };
        return Err(TerseError { code });
    }

    let out = unsafe {
        std::slice::from_raw_parts(compressed_values.data as *const u8, compressed_values.len).to_vec()
    };

    unsafe { capi::freeCompressedValues(&mut compressed_values) };

    Ok(out)
}


pub fn decompress(compressed: &[u8]) -> Result<Vec<f64>> {
    let compressed_values = CompressedValues {
        data: compressed.as_ptr() as *mut u8,
        len: compressed.len(),
    };

    let mut uncompressed_values = UncompressedValues {
        data: std::ptr::null(),
        len: 0,
    };

    let code = unsafe { capi::decompress(compressed_values, &mut uncompressed_values) };

    if code != 0 {
        unsafe { capi::freeUncompressedValues(&mut uncompressed_values) };
        return Err(TerseError { code });
    }

    let out = unsafe {
        let slice = std::slice::from_raw_parts(uncompressed_values.data, uncompressed_values.len);
        slice.to_vec()
    };

    unsafe { capi::freeUncompressedValues(&mut uncompressed_values) };

    Ok(out)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_loads_tersets_without_error() {
        let compressed_values = compress(
            &[10.0, 20.0, 30.0, 40.0, 50.0],
            Method::SwingFilter,
            r#"{ "abs_error_bound": 0.0 }"#,
        ).unwrap();

        let uncompressed_values = decompress(&compressed_values).unwrap();
        assert_eq!(uncompressed_values, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    }

}
