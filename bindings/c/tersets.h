// Copyright 2024 TerseTS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TERSETS_H
#define TERSETS_H

#include <stdint.h>
#include <stddef.h> // for size_t

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

/// A pointer to uncompressed values and the number of values.
struct UncompressedValues {
    const double* data;
    uintptr_t len;
};

/// A pointer to compressed values and the number of bytes.
struct CompressedValues {
    const uint8_t* data;
    uintptr_t len;
};

/// The type of error bound used in compression algorithms.
enum ErrorBoundType {
    TERSETS_ERROR_BOUND_ABSOLUTE,
    TERSETS_ERROR_BOUND_RELATIVE,
};

/// Cost function used in line simplification algorithms.
enum CostFunction{
    TERSETS_ERROR_BOUND_RMSE,
    TERSETS_ERROR_BOUND_LINF,
};

/// Error codes returned by TerseTS compression and decompression functions.
enum TerseTSError {
    TERSETS_ERROR_UNKNOWN_METHOD         = 1,
    TERSETS_ERROR_UNSUPPORTED_INPUT      = 2,
    TERSETS_ERROR_UNSUPPORTED_ERROR_BOUND= 3,
    TERSETS_ERROR_UNSUPPORTED_PARAMETERS = 4,
    TERSETS_ERROR_ITEM_NOT_FOUND         = 5,
    TERSETS_ERROR_OUT_OF_MEMORY          = 6,
    TERSETS_ERROR_EMPTY_CONVEX_HULL      = 7,
    TERSETS_ERROR_EMPTY_QUEUE            = 8,
};

/// Configuration to use for compression and decompression.
/// - `method`: the compression algorithm (enum value).
/// - `parameters`: a pointer to a method-specific struct 
//     (e.g., `FunctionalParams`) or a `BasicParams`. 
struct Configuration {
    uint8_t method;
    const void* parameters;
};

// -----------------------------------------------------------------------------
// Parameter Structs
// -----------------------------------------------------------------------------

/// Basic fallback parameters.
struct BasicParams {
    float error_bound;
};

/// Histogram-based compression parameters.
struct HistogramParams {
    size_t maximum_buckets;
};

/// Functional approximation parameters (e.g. PMC, SWING).
struct FunctionalParams {
    enum ErrorBoundType error_bound_type;
    float error_bound;
};

/// Line simplification parameters (e.g. Visvalingam-Whyatt).
struct LineSimplificationParams {
    enum ErrorBoundType error_bound_type;
    float error_bound;
};

// -----------------------------------------------------------------------------
// API Functions
// -----------------------------------------------------------------------------

/// Compress `uncompressed_values` into `compressed_values` using `configuration`.
/// Returns 0 on success, or a `TerseTSError` on failure.
int32_t compress(
    struct UncompressedValues uncompressed_values,
    struct CompressedValues* compressed_values,
    struct Configuration configuration);

/// Decompress `compressed_values` into `uncompressed_values`.
/// Returns 0 on success, or a `TerseTSError` on failure.
int32_t decompress(
    struct CompressedValues compressed_values,
    struct UncompressedValues* uncompressed_values);

/// Returns a human-readable error message for a given `TerseTSError` code.
const char* tersets_strerror(int32_t code);

#ifdef __cplusplus
}
#endif

#endif // TERSETS_H
