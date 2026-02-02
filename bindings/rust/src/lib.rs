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

//! Python bindings for the TerseTS library.

mod capi;

use std::ffi::CString;
use std::ptr;

use capi::{CompressedValues, UncompressedValues};

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

impl From<Method> for u8 {
    fn from(method: Method) -> Self {
        match method {
            Method::PoorMansCompressionMidrange => 0,
            Method::PoorMansCompressionMean => 1,
            Method::SwingFilter => 2,
            Method::SwingFilterDisconnected => 3,
            Method::SlideFilter => 4,
            Method::SimPiece => 5,
            Method::PiecewiseConstantHistogram => 6,
            Method::PiecewiseLinearHistogram => 7,
            Method::ABCLinearApproximation => 8,
            Method::VisvalingamWhyatt => 9,
            Method::SlidingWindow => 10,
            Method::BottomUp => 11,
            Method::MixPiece => 12,
            Method::BitPackedQuantization => 13,
            Method::RunLengthEncoding => 14,
            Method::NonLinearApproximation => 15,
            Method::SerfQT => 16,
        }
    }
}

pub fn compress(uncompressed_values: &[f64], method: Method, configuration: &str) -> Vec<u8> {
    let uncompressed_values = UncompressedValues {
        data: uncompressed_values.as_ptr(),
        len: uncompressed_values.len(),
    };

    let mut compressed_values = CompressedValues {
        data: ptr::null_mut(),
        len: 0,
    };

    // TODO: Replace unwrap() with a proper error.
    let configuration = CString::new(configuration).unwrap();

    let tersets_error = unsafe {
        capi::compress(
            uncompressed_values,
            &mut compressed_values,
            method.into(),
            configuration.as_ptr(),
        )
    };

    vec![]
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
        );
        //TODO: Include decompress() and check values are the same.
    }
}
