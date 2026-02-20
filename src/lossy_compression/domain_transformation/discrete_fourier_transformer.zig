// Copyright 2025 TerseTS Contributors
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

//! This module implements Discrete Fourier Transform (DFT) using the PocketFFT library implemented at:
//! https://gitlab.mpcdf.mpg.de/mtr/pocketfft.
//! It provides functions to compress and decompress data using DFT by retaining a specified number of
//! frequency coefficients based on their magnitudes. The compression function transforms the input data
//! into the frequency domain, selects the top-K coefficients, and stores them along with necessary metadata.
//! The decompression function reconstructs the original data approximately by performing an inverse DFT
//! using the retained coefficients. The module also includes an `extract` function to retrieve the preserved
//! frequency coefficients from the compressed data without performing a full decompression. Additonally, a
//! rebuild function is provided to reconstruct the signal from the extracted coefficients. Error handling is
//! implemented to manage invalid configurations and corrupted compressed data.

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;

const tersets = @import("../../tersets.zig");
const configuration = @import("../../configuration.zig");
const shared_functions = @import("../../utilities/shared_functions.zig");
const tester = @import("../../tester.zig");

const Error = tersets.Error;

/// Opaque type for PocketFFT plan.
pub const rfft_plan = ?*opaque {};

/// FFI declarations for PocketFFT C functions.
extern "c" fn make_rfft_plan(length: usize) rfft_plan;
extern "c" fn destroy_rfft_plan(plan: rfft_plan) void;
extern "c" fn rfft_forward(plan: rfft_plan, values: [*]f64, scale: f64) c_int;
extern "c" fn rfft_backward(plan: rfft_plan, values: [*]f64, scale: f64) c_int;
extern "c" fn rfft_length(plan: rfft_plan) usize;

/// Compresses the `uncompressed_values` using Discrete Fourier Transform (DFT) retaining
/// `number_of_coefficients` largest magnitude coefficients as specified in the
/// `method_configuration`. The compressed data is composed of the number of coefficients,
/// original length, DC coefficient, and the selected frequency coefficients. If an error
/// occurs, it is returned.
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    const number_of_bins: usize = uncompressed_values.len / 2 + 1;

    // Validate input length. The maximum length is constrained by the header format.
    // This is a practical limit to ~ 4 billion samples, which should be sufficient.
    if (uncompressed_values.len > 0xFFFFFFFF)
        return Error.UnsupportedInput;

    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.DomainTransformation,
        method_configuration,
    );
    const number_of_coefficients: u32 = parsed_configuration.number_of_coefficients;

    if (number_of_coefficients > number_of_bins)
        return Error.InvalidConfiguration;

    // PocketFFT works in-place.
    const pocketfft_buffer = try allocator.alloc(f64, uncompressed_values.len);
    defer allocator.free(pocketfft_buffer);

    // Copy input values to PocketFFT buffer.
    @memcpy(pocketfft_buffer, uncompressed_values);

    // Create FFT plan.
    const plan = make_rfft_plan(uncompressed_values.len);
    defer destroy_rfft_plan(plan);

    if (plan == null) return Error.UnsupportedInput;

    // Perform forward FFT.
    if (rfft_forward(plan, pocketfft_buffer.ptr, 1.0) != 0)
        return Error.UnsupportedInput;

    // DC coefficient at the first index always stored separately.
    const dc_real: f64 = pocketfft_buffer[0];

    // Collect candidate coefficients (excluding DC).
    const CandidateCoefficient = struct {
        index: u64,
        real_value: f64,
        imaginary_value: f64,
        magnitude_square: f64,
    };

    // There are number_of_bins - 1 candidate coefficients (excluding DC).
    var candidates = try allocator.alloc(CandidateCoefficient, number_of_bins - 1);
    defer allocator.free(candidates);

    // Depending on even/odd length, determine how many bins to process.
    // For even length, the Nyquist frequency bin is real-only and stored separately.
    const run_until: usize =
        if (uncompressed_values.len % 2 == 0)
            number_of_bins - 1
        else
            number_of_bins;

    // Populate candidates array.
    var j: usize = 0;
    for (1..run_until) |k| {
        const real = pocketfft_buffer[2 * k - 1];
        const imaginary = pocketfft_buffer[2 * k];
        candidates[j] = .{
            .index = k,
            .real_value = real,
            .imaginary_value = imaginary,
            .magnitude_square = real * real + imaginary * imaginary,
        };
        j += 1;
    }

    if (uncompressed_values.len % 2 == 0) {
        // Nyquist bin k = half, stored at buffer[N-1], real-only.
        const real_nyquist = pocketfft_buffer[uncompressed_values.len - 1];
        candidates[j] = .{
            .index = run_until,
            .real_value = real_nyquist,
            .imaginary_value = 0.0,
            .magnitude_square = real_nyquist * real_nyquist,
        };
    }

    // Sort by descending magnitude_square.
    mem.sort(CandidateCoefficient, candidates, {}, struct {
        fn highest_first(_: void, a: CandidateCoefficient, b: CandidateCoefficient) bool {
            return a.magnitude_square > b.magnitude_square;
        }
    }.highest_first);

    // Store header of the compressed data.
    // 1) Number of retained coefficients (u32).
    // 2) Original length (u32).
    // 3) DC coefficient (f64).
    try shared_functions.appendValue(allocator, u32, number_of_coefficients, compressed_values);
    try shared_functions.appendValue(
        allocator,
        u32,
        @as(u32, @intCast(uncompressed_values.len)),
        compressed_values,
    );
    try shared_functions.appendValue(
        allocator,
        f64,
        dc_real,
        compressed_values,
    );

    // Top-K non-DC coefficients.
    for (0..number_of_coefficients - 1) |i| {
        const candidate = candidates[i];
        try shared_functions.appendValue(
            allocator,
            u64,
            candidate.index,
            compressed_values,
        );
        try shared_functions.appendValue(
            allocator,
            f64,
            candidate.real_value,
            compressed_values,
        );
        try shared_functions.appendValue(
            allocator,
            f64,
            candidate.imaginary_value,
            compressed_values,
        );
    }
}

/// Decompresses the `compressed_values` using Discrete Fourier Transform (DFT),
/// reconstructing the original signal approximately by retaining only the
/// specified number of frequency coefficients. The decompressed values are
/// appended to `decompressed_values`. If an error occurs, it is returned.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    if (compressed_values.len < 16)
        return Error.CorruptedCompressedData; // u32 + u64 + f64.

    var offset: usize = 0;

    // Read header information.
    const preserve_top_coefficients =
        try shared_functions.readOffsetValue(u32, compressed_values, &offset);
    const N: u32 =
        try shared_functions.readOffsetValue(u32, compressed_values, &offset);

    // Just in case, validate N. TerseTS API should prevent any array with length less than 2
    // from being compressed, but we check here to be safe.
    if (N < 2)
        return Error.CorruptedCompressedData;

    const number_of_bins: usize = @intCast(N / 2 + 1);

    // Validate number of coefficients to preserve.
    if (preserve_top_coefficients == 0 or preserve_top_coefficients > number_of_bins)
        return Error.CorruptedCompressedData;

    // Read DC coefficient.
    const dc_real: f64 =
        try shared_functions.readOffsetValue(f64, compressed_values, &offset);

    // Prepare PocketFFT buffer.
    var pocketfft_buffer = try allocator.alloc(f64, N);
    defer allocator.free(pocketfft_buffer);
    @memset(pocketfft_buffer, 0.0);

    pocketfft_buffer[0] = dc_real;
    // Read preserved coefficients.
    for (0..preserve_top_coefficients - 1) |_| {
        const index =
            try shared_functions.readOffsetValue(u64, compressed_values, &offset);
        const real =
            try shared_functions.readOffsetValue(f64, compressed_values, &offset);
        const imaginary =
            try shared_functions.readOffsetValue(f64, compressed_values, &offset);

        if (index == 0 or index >= number_of_bins)
            return Error.CorruptedCompressedData;

        if ((N % 2 == 0) and (index == number_of_bins - 1)) {
            // Nyquist: real-only.
            pocketfft_buffer[N - 1] = real;
        } else {
            pocketfft_buffer[2 * index - 1] = real;
            pocketfft_buffer[2 * index] = imaginary;
        }
    }

    // Create FFT plan.
    const plan = make_rfft_plan(N);
    if (plan == null) return Error.UnsupportedInput;
    defer destroy_rfft_plan(plan);

    // Perform inverse FFT.
    const scale: f64 = 1.0 / @as(f64, @floatFromInt(N));
    if (rfft_backward(plan, pocketfft_buffer.ptr, scale) != 0)
        return Error.UnsupportedInput;

    // Append decompressed values.
    try decompressed_values.ensureTotalCapacity(allocator, N);
    for (0..N) |i_val| {
        try decompressed_values.append(allocator, pocketfft_buffer[i_val]);
    }
}

/// Extracts `indices` and `coefficients` from Discrete Fourier Transform's `compressed_values`.
/// The `compressed_values` encodes: real and imaginary parts of the preserved coefficients, and
/// their corresponding indices. The first 16 bytes of `compressed_values` are reserved in the
/// `indices` array for the header information (number of coefficients, original length, DC coefficient).
/// The `indices` ArrayList stores the indices of the preserved coefficients, and the `coefficients`
/// ArrayList stores the real and imaginary parts of the preserved coefficients in an interleaved manner
/// (real, imaginary, real, imaginary, ...). Any loss of information on the indices, for example,
/// incorrect header information, or missing indices, can lead to incorrect reconstruction of the signal
/// during decompression. Only structural checks are performed. The caller must ensure semantic validity.
/// If the compressed stream does not follow the expected representation, `Error.CorruptedCompressedData`
/// is returned. The `allocator` handles the memory allocations of the output arrays.
/// Allocation errors are propagated.
pub fn extract(
    allocator: Allocator,
    compressed_values: []const u8,
    indices: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    if (compressed_values.len < 16)
        return Error.CorruptedCompressedData; // u32 + u64 + f64.

    var offset: usize = 0;

    // Read header information.
    const preserve_top_coefficients: u64 =
        @intCast(try shared_functions.readOffsetValue(u32, compressed_values, &offset));
    const N: u64 =
        @intCast(try shared_functions.readOffsetValue(u32, compressed_values, &offset));

    // Store header information in the indices array for reference during reconstruction.
    try indices.append(allocator, preserve_top_coefficients);
    try indices.append(allocator, N);

    // Read DC coefficient and store it in the coefficients array for reference during reconstruction.
    const dc_real: f64 =
        try shared_functions.readOffsetValue(f64, compressed_values, &offset);

    try coefficients.append(allocator, dc_real);

    // Just in case, validate N. TerseTS API should prevent any array with length less than 2
    // from being compressed, but we check here to be safe.
    if (N < 2)
        return Error.CorruptedCompressedData;

    const number_of_bins: usize = @intCast(N / 2 + 1);

    // Validate number of coefficients to preserve.
    if (preserve_top_coefficients == 0 or preserve_top_coefficients > number_of_bins)
        return Error.CorruptedCompressedData;

    // Skip DC coefficient.
    offset += 8;

    // Read preserved coefficients.
    for (0..preserve_top_coefficients - 1) |_| {
        const index =
            try shared_functions.readOffsetValue(u64, compressed_values, &offset);
        const real =
            try shared_functions.readOffsetValue(f64, compressed_values, &offset);
        const imaginary =
            try shared_functions.readOffsetValue(f64, compressed_values, &offset);

        if (index == 0 or index >= number_of_bins)
            return Error.CorruptedCompressedData;

        try indices.append(allocator, index);
        try coefficients.append(allocator, real);
        try coefficients.append(allocator, imaginary);
    }
}

/// Rebuilds Discrete Fourier Transform's `compressed_values` from the given `indices` and
/// `coefficients`. The encoding consists of top coefficients to preserve, original length,
/// DC coefficient, and the indices and values of the preserved coefficients. The `indices`
/// ArrayList is expected to contain the indices of the preserved coefficients, and the
/// `coefficients` ArrayList is expected to contain the real and imaginary parts of the
/// preserved coefficients in an interleaved manner (real, imaginary, real, imaginary, ...).
/// Any loss or misalignment of indices information, for example, wrong number of elements,
/// or top coefficient count can lead to failures during decompression. The function checks
/// for structural consistency and returns `Error.CorruptedCompressedData` for malformed input.
/// The `allocator` handles the memory allocations of the output arrays. Allocation errors are propagated.
pub fn rebuild(
    allocator: Allocator,
    indices: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Must contain at least shift_amount and number_of_segments.
    if (indices.len < 1 or coefficients.len < 1)
        return Error.CorruptedCompressedData;

    // Coefficients must be at least 2 times the number of indices (real and imaginary parts).
    // We remove the header from the indices count, since the first two indices are reserved for
    // the number of coefficients and original length.
    if (coefficients.len * 2 < indices.len - 2) {
        return Error.CorruptedCompressedData;
    }

    // Store header of the compressed data.
    // 1) Number of retained coefficients (u32).
    // 2) Original length (u32).
    // 3) DC coefficient (f64).
    const preserve_top_coefficients: u64 = indices[0];
    const N: u64 = indices[1];
    const dc_real: f64 = coefficients[0];

    try shared_functions.appendValue(
        allocator,
        u32,
        @as(u32, @intCast(preserve_top_coefficients)),
        compressed_values,
    );
    try shared_functions.appendValue(
        allocator,
        u32,
        @as(u32, @intCast(N)),
        compressed_values,
    );
    try shared_functions.appendValue(
        allocator,
        f64,
        dc_real,
        compressed_values,
    );

    // Append preserved coefficients.
    for (2..indices.len) |i| {
        const index = indices[i];
        const real = coefficients[2 * i - 1];
        const imaginary = coefficients[2 * i];

        try shared_functions.appendValue(
            allocator,
            u64,
            index,
            compressed_values,
        );
        try shared_functions.appendValue(
            allocator,
            f64,
            real,
            compressed_values,
        );
        try shared_functions.appendValue(
            allocator,
            f64,
            imaginary,
            compressed_values,
        );
    }
}

test "fft compression round-trip full reconstruction with all coefficients preserved and small values" {
    // This test validates that compressing and then decompressing a random array of values
    // results in an approximation of the original values within a reasonable error margin,
    // when all coefficients are preserved. This serves as a basic correctness check for the
    // DFT-based compression and decompression implementation.
    // Note: since we are preserving all coefficients, the decompressed values should be very
    // close to the original values, with only minor differences due to floating-point precision.
    // DFT cannot be set 100% lossless compression due to the nature of floating-point arithmetic.
    const allocator = std.testing.allocator;

    var uncompressed = ArrayList(f64).empty;
    defer uncompressed.deinit(allocator);

    // Generate random values between -1000 and 1000 to ensure a reasonable range of magnitudes.
    // Generating values between -tester.max_test_values and tester.max_test_values can lead to
    // very large magnitudes in the frequency domain, which can amplify floating-point precision
    // issues and result in larger errors after decompression. By constraining the input values to
    // a smaller range, we can mitigate this issue and ensure that the test focuses on validating
    // the correctness of the compression and decompression logic rather than being affected by
    // extreme values.
    try tester.generateBoundedRandomValues(
        allocator,
        &uncompressed,
        -1e3,
        1e3,
        null,
    );
    const N = uncompressed.items.len;

    // Preserve all coefficients for this test to validate the round-trip reconstruction.
    const preserve_top_coefficients = N / 2 + 1; // number_of_bins + 1.

    var compressed_values = std.ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"number_of_coefficients\": {}}}",
        .{preserve_top_coefficients},
    );
    defer allocator.free(method_configuration);

    try compress(
        allocator,
        uncompressed.items,
        &compressed_values,
        method_configuration,
    );

    var decompressed = std.ArrayList(f64).empty;
    defer decompressed.deinit(allocator);

    try decompress(
        allocator,
        compressed_values.items,
        &decompressed,
    );

    try std.testing.expectEqual(uncompressed.items.len, decompressed.items.len);

    // Validate that each decompressed value is approximately equal to the original value within
    // small error margin. The value 1e-8 is chosen as a reasonable threshold for floating-point
    // precision errors in this context. Since we are preserving all coefficients, the decompressed
    // values should be very close. There is no strict theoretical guarantee for the exact error margin.
    // Thus, 1e-8 is an empirically chosen value that balances the need to account for
    // floating-point precision issues while still ensuring that the decompressed values are very
    // close to the original values.
    for (0..N) |i| {
        try std.testing.expectApproxEqRel(
            decompressed.items[i],
            uncompressed.items[i],
            1e-8,
        );
    }
}

test "fft compression round-trip full reconstruction with all coefficients preserved and very small values" {
    // This test validates that compressing and then decompressing a random array of values
    // results in an approximation of the original values within a reasonable error margin,
    // when all coefficients are preserved. This serves as a basic correctness check for the
    // DFT-based compression and decompression implementation.
    // Note: since we are preserving all coefficients, the decompressed values should be very
    // close to the original values, with only minor differences due to floating-point precision.
    // DFT cannot be set 100% lossless compression due to the nature of floating-point arithmetic.
    const allocator = std.testing.allocator;

    var uncompressed = ArrayList(f64).empty;
    defer uncompressed.deinit(allocator);

    // Generate random values between -1 and 1 to ensure a reasonable range of magnitudes.
    // Generating values between -tester.max_test_values and tester.max_test_values can lead to
    // very large magnitudes in the frequency domain, which can amplify floating-point precision
    // issues and result in larger errors after decompression. By constraining the input values to
    // a smaller range, we can mitigate this issue and ensure that the test focuses on validating
    // the correctness of the compression and decompression logic rather than being affected by
    // extreme values.
    try tester.generateBoundedRandomValues(
        allocator,
        &uncompressed,
        -1,
        1,
        null,
    );
    const N = uncompressed.items.len;

    // Preserve all coefficients for this test to validate the round-trip reconstruction.
    const preserve_top_coefficients = N / 2 + 1; // number_of_bins + 1.

    var compressed_values = std.ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"number_of_coefficients\": {}}}",
        .{preserve_top_coefficients},
    );
    defer allocator.free(method_configuration);

    try compress(
        allocator,
        uncompressed.items,
        &compressed_values,
        method_configuration,
    );

    var decompressed = std.ArrayList(f64).empty;
    defer decompressed.deinit(allocator);

    try decompress(
        allocator,
        compressed_values.items,
        &decompressed,
    );

    try std.testing.expectEqual(uncompressed.items.len, decompressed.items.len);

    // Validate that each decompressed value is approximately equal to the original value within
    // small error margin. The value 1e-10 is chosen as a reasonable threshold for floating-point
    // precision errors in this context. Since we are preserving all coefficients, the decompressed
    // values should be very close. There is no strict theoretical guarantee for the exact error margin.
    // Thus, 1e-10 is an empirically chosen value that balances the need to account for
    // floating-point precision issues while still ensuring that the decompressed values are very
    // close to the original values.
    for (0..N) |i| {
        try std.testing.expectApproxEqRel(
            decompressed.items[i],
            uncompressed.items[i],
            1e-10,
        );
    }
}

test "fft compression partial reconstruction preserves mean error" {
    const allocator = std.testing.allocator;

    // Generate the values between -1 and 1 to validate mean error after.
    var uncompressed = ArrayList(f64).empty;
    defer uncompressed.deinit(allocator);

    try tester.generateBoundedRandomValues(
        allocator,
        &uncompressed,
        -1,
        1,
        null,
    );

    const N: usize = uncompressed.items.len;

    var compressed = std.ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    const preserve_top_coefficients: usize = tester.generateBoundRandomInteger(
        usize,
        3,
        N / 2 - 1,
        null,
    );

    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"number_of_coefficients\": {}}}",
        .{preserve_top_coefficients},
    );
    defer allocator.free(method_configuration);

    try compress(
        allocator,
        uncompressed.items,
        &compressed,
        method_configuration,
    );

    var decompressed = std.ArrayList(f64).empty;
    defer decompressed.deinit(allocator);

    try decompress(
        allocator,
        compressed.items,
        &decompressed,
    );

    try std.testing.expectEqual(uncompressed.items.len, decompressed.items.len);

    var mean_error: f64 = 0;
    const size: f64 = @floatFromInt(N);
    for (uncompressed.items, 0..) |original_value, i| {
        const decompressed_value = decompressed.items[i];
        mean_error += @abs(original_value - decompressed_value) / size;
    }

    // Since all values are between -1 and 1, the mean_error should be less than 1.
    try std.testing.expect(mean_error < 1);
}

test "fft compression only returns known mean value" {
    const allocator = std.testing.allocator;

    const uncompressed = [_]f64{ 3, 6, 9, 12, 15, 18 };
    const N = uncompressed.len;
    const mean = 10.5; // Known value given the input.

    var compressed = std.ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    // Retain only the DC coefficient.
    const method_configuration = "{\"number_of_coefficients\": 1}";

    try compress(allocator, uncompressed[0..], &compressed, method_configuration);

    var decompressed = std.ArrayList(f64).empty;
    defer decompressed.deinit(allocator);

    try decompress(allocator, compressed.items, &decompressed);

    try std.testing.expectEqual(N, decompressed.items.len);

    for (decompressed.items) |value| {
        try std.testing.expectEqual(value, mean);
    }
}
