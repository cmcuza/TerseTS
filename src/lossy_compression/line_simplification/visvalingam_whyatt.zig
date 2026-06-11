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

//! Implementation of the Visvalingam-Whyatt line simplification algorithm from the paper
//! "Visvalingam, M.; Whyatt, J. D. Line generalisation by repeated elimination of points.
//! The Cartographic Journal. 30 (1): 46–51, 1993.
//! https://doi.org/10.1179/000870493786962263".

const std = @import("std");
const mem = std.mem;
const math = std.math;
const testing = std.testing;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

const tersets = @import("../../tersets.zig");
const configuration = @import("../../configuration.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const shared_structs = @import("../../utilities/shared_structs.zig");
const shared_functions = @import("../../utilities/shared_functions.zig");

const DiscretePoint = shared_structs.DiscretePoint;
const LinearFunction = shared_structs.LinearFunction;
const Segment = shared_structs.Segment;

const tester = @import("../../tester.zig");

const extractors = @import("../../utilities/extractors.zig");
const rebuilders = @import("../../utilities/rebuilders.zig");
const IndexedPriorityQueue = @import("../../utilities/indexed_priority_queue.zig").IndexedPriorityQueue;

/// Compress `uncompressed_values` using "Visvalingam-Whyatt" simplification algorithm by keeping
/// points whose effective area is greater than the `error_bound`. The function writes the result
/// to `compressed_values`. The `allocator` is used to allocate memory for the temporary
/// simplification state and the `method_configuration` parser. The `method_configuration`
/// is expected to be of `AreaUnderCurveError` type otherwise an `InvalidConfiguration` error is return.
/// If any other error occurs during the execution of the method, it is returned.
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.AreaUnderCurveError,
        method_configuration,
    );

    const error_bound: f32 = parsed_configuration.area_under_curve_error;

    // If we have 2 or fewer points, store them without compression.
    if (uncompressed_values.len <= 2) {
        try shared_functions.appendValue(allocator, f64, uncompressed_values[0], compressed_values);
        try shared_functions.appendValue(allocator, f64, uncompressed_values[1], compressed_values);
        try shared_functions.appendValue(allocator, usize, 1, compressed_values);
        return;
    }

    const point_count = uncompressed_values.len;

    var heap = try IndexedPriorityQueue(f64, compareArea).init(allocator, point_count);
    defer heap.deinit();

    const left_points = try allocator.alloc(usize, point_count);
    defer allocator.free(left_points);
    const right_points = try allocator.alloc(usize, point_count);
    defer allocator.free(right_points);

    for (0..point_count) |i| {
        left_points[i] = if (i == 0) 0 else i - 1;
        right_points[i] = i + 1;
        const area = if (i == 0 or i == point_count - 1)
            math.inf(f64)
        else
            calculateAreaFromValues(uncompressed_values, i - 1, i, i + 1);
        try heap.add(i, area);
    }

    var remaining_points = point_count;

    // Main simplification loop.
    while (heap.count() > 2) {
        // Get the point with the smallest area.
        const min_point = try heap.peek();

        // The area is greater than the error bound.
        if (min_point.priority >= error_bound) {
            break;
        }

        // Now is safe to remove the point with the smallest area.
        _ = try heap.pop();
        remaining_points -= 1;

        const left_point = left_points[min_point.index];
        const right_point = right_points[min_point.index];

        right_points[left_point] = right_point;
        left_points[right_point] = left_point;

        // Update areas of left and right neighbors. Endpoints stay at infinity and cannot be removed.
        try updateNeighborArea(
            &heap,
            left_point,
            left_points,
            right_points,
            uncompressed_values,
        );
        try updateNeighborArea(
            &heap,
            right_point,
            left_points,
            right_points,
            uncompressed_values,
        );
    }

    // Output compressed series: first point, then (value, index) pairs.
    try compressed_values.ensureTotalCapacity(allocator, compressed_values.items.len + 8 + (remaining_points - 1) * 16);
    try shared_functions.appendValue(allocator, f64, uncompressed_values[0], compressed_values);
    var point_index = right_points[0];
    while (point_index < point_count) : (point_index = right_points[point_index]) {
        try shared_functions.appendValue(allocator, f64, uncompressed_values[point_index], compressed_values);
        try shared_functions.appendValue(allocator, usize, point_index, compressed_values);
    }

    return;
}

/// Decompress `compressed_values` produced by "Visvalingam-Whyatt" and write the
/// result to `decompressed_values`. If an error occurs it is returned.
pub fn decompress(allocator: Allocator, compressed_values: []const u8, decompressed_values: *ArrayList(f64)) Error!void {
    // The compressed representation is composed of two values after getting the first since all
    // segments are connected. Therefore, the condition checks that after the first value, the rest
    // of the values are in pairs (value, index) and that they are all of type 64-bit float.
    if ((compressed_values.len - 8) % 16 != 0) return Error.UnsupportedInput;

    const compressed_lines_and_index = mem.bytesAsSlice(f64, compressed_values);

    var index: usize = 0;

    // Extract the start point from the compressed representation.
    var start_point: DiscretePoint = .{ .index = 0, .value = compressed_lines_and_index[0] };
    try decompressed_values.append(allocator, start_point.value);

    // We need to create a segment for the linear function.
    var slope: f64 = undefined;
    var intercept: f64 = undefined;
    while (index < compressed_lines_and_index.len - 1) : (index += 2) {
        // index + 1 is the end value and index + 2 is the end time.
        const end_point: DiscretePoint = .{
            .index = @as(usize, @bitCast(compressed_lines_and_index[index + 2])),
            .value = compressed_lines_and_index[index + 1],
        };

        if (start_point.index + 1 < end_point.index) {
            // Create the linear approximation for the current segment.
            const duration: f64 = @floatFromInt(end_point.index - start_point.index);
            slope = (end_point.value - start_point.value) / duration;
            intercept = start_point.value - slope *
                @as(f64, @floatFromInt(start_point.index));

            var current_index: usize = start_point.index + 1;
            // Interpolate the values between the start and end points of the current segment.
            while (current_index < end_point.index) : (current_index += 1) {
                const y: f64 = slope * @as(f64, @floatFromInt(current_index)) + intercept;
                try decompressed_values.append(allocator, y);
            }
        }
        try decompressed_values.append(allocator, end_point.value);

        // The start point of the next segment is the end point of the current segment.
        start_point = end_point;
    }
}

/// Extracts `indices` and `coefficients` from Visvalingam-Whyatt's `compressed_values`. The binary
/// representation follows the same pattern as SwingFilter, so this function calls `extractSwing`.
/// All structural and corruption checks are performed by the delegated function. Any loss of
/// index information can lead to failures during later decompression. The `allocator` handles
/// the memory of the output arrays. Allocation errors are propagated.
pub fn extract(
    allocator: Allocator,
    compressed_values: []const u8,
    indices: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // Delegate to CoefficientIndexTuplesWithStartCoefficient extractor.
    // VisvalingamWhyatt uses the same representation as SwingFilter.
    try extractors.extractCoefficientIndexTuplesWithStartCoefficient(
        allocator,
        compressed_values,
        indices,
        coefficients,
    );
}

/// Rebuilds Visvalingam-Whyatt's `compressed_values` from the provided `indices` and `coefficients`.
/// The representation matches SwingFilter, so the function delegates to `rebuildSwing`. All format
/// validation and corruption checks are performed by that routine. Any loss or misalignment of
/// indices may cause failures when decoding the rebuilt representation. The `allocator` handles
/// the memory of the output arrays. Allocation errors are propagated.
pub fn rebuild(
    allocator: Allocator,
    indices: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Delegate to CoefficientIndexTuplesWithStartCoefficient extractor.
    // VisvalingamWhyatt uses the same representation as SwingFilter.
    try rebuilders.rebuildCoefficientIndexTuplesWithStartCoefficient(
        allocator,
        indices,
        coefficients,
        compressed_values,
    );
}

/// Return the absolute area of the triangle defined by three points. The points are represented as
/// `DiscretePoint` structs, which contain an `index` and a `value`. The `index` represents the
/// position of the point in the original uncompressed series, while the `value` represents the value
/// of the point. The function calculates the area using the formula for the area of a triangle given
/// by three points in a 2D plane, where the x-coordinate is given by the `index` and the y-coordinate
/// is given by the `value`. The function returns the absolute value of the area.
fn calculateArea(left_point: DiscretePoint, central_point: DiscretePoint, right_point: DiscretePoint) f64 {
    const x1: f64 = @floatFromInt(left_point.index);
    const y1: f64 = left_point.value;
    const x2: f64 = @floatFromInt(central_point.index);
    const y2: f64 = central_point.value;
    const x3: f64 = @floatFromInt(right_point.index);
    const y3: f64 = right_point.value;

    return @abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0);
}

/// Return the absolute area of the triangle defined by three points whose indices are `left_index`,
/// `center_index` and `right_index` and whose values are in the `values` array. The `values` array
/// is expected to contain the values of the points in the same order as their indices. The function
/// does not perform any check on the validity of the indices or the values, so it is the caller's
/// responsibility to ensure that they are correct.
fn calculateAreaFromValues(values: []const f64, left_index: usize, center_index: usize, right_index: usize) f64 {
    const x1: f64 = @floatFromInt(left_index);
    const y1: f64 = values[left_index];
    const x2: f64 = @floatFromInt(center_index);
    const y2: f64 = values[center_index];
    const x3: f64 = @floatFromInt(right_index);
    const y3: f64 = values[right_index];

    return @abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0);
}

/// Compare two areas for the priority queue. The function returns `.lt` if `area_1` is less
/// than `area_2`, `.eq` if they are equal and `.gt` if `area_1` is greater than `area_2`.
fn compareArea(area_1: f64, area_2: f64) math.Order {
    return math.order(area_1, area_2);
}

/// Update the area of `neighbor_index` in the `heap` based on its current neighbors.
/// `left_points` and `right_points` provide the current linked-list style adjacency between points.
/// `uncompressed_values` are used to compute the triangle area for the updated neighborhood.
fn updateNeighborArea(
    heap: *IndexedPriorityQueue(f64, compareArea),
    neighbor_index: usize,
    left_points: []const usize,
    right_points: []const usize,
    uncompressed_values: []const f64,
) !void {
    var area = math.inf(f64);
    if (neighbor_index > 0 and neighbor_index < uncompressed_values.len - 1) {
        area = calculateAreaFromValues(
            uncompressed_values,
            left_points[neighbor_index],
            neighbor_index,
            right_points[neighbor_index],
        );
    }

    try heap.update(neighbor_index, area);
}

/// Test if the area of all the triangles defined by three points contained in `values`
/// is within the `error_bound`.
pub fn testAreaWithinErrorBound(
    values: []const f64,
    error_bound: f32,
) !void {
    // At least three points are needed to form a triangle.
    if (values.len < 3) return;

    // Calculate the area formed by each triangle.
    for (1..values.len - 1) |i| {
        const area = calculateArea(
            DiscretePoint{ .index = i - 1, .value = values[i - 1] },
            DiscretePoint{ .index = i, .value = values[i] },
            DiscretePoint{ .index = i + 1, .value = values[i + 1] },
        );
        try testing.expect(area <= error_bound);
    }
}

test "vw compress and decompress with zero error bound" {
    // Initialize a random number generator.
    const seed: u64 = @bitCast(tester.milliTimestamp());
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    const allocator = testing.allocator;

    // Output buffer.
    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);
    const error_bound: f32 = 0.0;

    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, 0, 1000000, random);

    const method_configuration =
        \\ {"area_under_curve_error": 0.0}
    ;

    // Call the compress and decompress functions.
    try compress(
        allocator,
        uncompressed_values.items,
        &compressed_values,
        method_configuration,
    );
    try decompress(allocator, compressed_values.items, &decompressed_values);

    try testing.expect(shared_functions.isWithinErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        error_bound,
    ));
}

test "vw compress and compress with known result" {
    const allocator = testing.allocator;

    // Input data.
    const uncompressed_values: []const f64 = &[_]f64{ 1.0, 1.5, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0 };

    // Output buffer.
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    const method_configuration =
        \\ {"area_under_curve_error": 2.5}
    ;

    // Call the compress function.
    try compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );
    try decompress(allocator, compressed_values.items, &decompressed_values);

    // Check if the decompressed values have the same lenght as the compressed ones.
    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    // In theory, all triangles formed by the slices of removed points should be within the error otherwise the
    // point cannot be removed. In this case, the error is 2.5,  and the area of the triangles is always less than
    // this value except when removing the element in position 5. Therefore, the area of the triangles formed by the
    // slices of the removed points from 0..5 should be less than 2.5.
    try testAreaWithinErrorBound(uncompressed_values[0..6], 2.5);
}

test "vw compress recalculates areas for points next to endpoints" {
    const allocator = testing.allocator;

    const uncompressed_values: []const f64 = &[_]f64{ 1.0, 3.0, 4.0, 1.0 };
    const method_configuration =
        \\ {"area_under_curve_error": 2.5}
    ;

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    try compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );
    try decompress(allocator, compressed_values.items, &decompressed_values);

    const compressed_representation = mem.bytesAsSlice(f64, compressed_values.items);
    try testing.expectEqual(@as(usize, 5), compressed_representation.len);
    try testing.expectEqual(@as(usize, 2), @as(usize, @bitCast(compressed_representation[2])));
    try testing.expectEqual(@as(usize, 3), @as(usize, @bitCast(compressed_representation[4])));
    try testing.expectEqualSlices(f64, &[_]f64{ 1.0, 2.5, 4.0, 1.0 }, decompressed_values.items);
}

test "vw compress and compress with random data" {
    const random = tester.getDefaultRandomGenerator();

    const allocator = testing.allocator;

    // Output buffer.
    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);
    const error_bound: f32 = random.float(f32);

    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, 0, 1, random);

    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"area_under_curve_error\": {d}}}",
        .{error_bound},
    );
    defer allocator.free(method_configuration);

    // Call the compress function.
    try compress(
        allocator,
        uncompressed_values.items,
        &compressed_values,
        method_configuration,
    );
    try decompress(allocator, compressed_values.items, &decompressed_values);

    // Check if the decompressed values have the same lenght as the compressed ones.
    try testing.expectEqual(uncompressed_values.items.len, decompressed_values.items.len);

    // In theory, all triangles formed by the slices of removed points should be within the error otherwise the
    // point cannot be removed. In this case, the error bound is unknown as well as which points are finally
    // preserved in the compressed representation. Therefore, we need to used the compressed representation to access
    // each of the points preserved and their index `current_point_index`. Then, the area of the triangles formed by the
    // slices of the removed points from `previous_point_index`..`current_point_index` should be less than `error_bound`.
    const compressed_representation = mem.bytesAsSlice(f64, compressed_values.items);

    var index: usize = 0;
    var previous_point_index: usize = 0;
    while (index < compressed_representation.len - 1) : (index += 2) {
        const current_point_index = @as(usize, @bitCast(compressed_representation[index + 2]));

        // Check if the point is within the error bound.
        try testAreaWithinErrorBound(uncompressed_values.items[previous_point_index .. current_point_index + 1], error_bound);
        previous_point_index = current_point_index;
    }
}

test "check vw configuration parsing" {
    // Tests the configuration parsing and functionality of the `compress` function.
    // The test verifies that the provided configuration is correctly interpreted and
    // that the `configuration.AreaUnderCurveError` is expected in the function.
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 28.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"area_under_curve_error": 0.3}
    ;

    // The configuration is properly defined. No error expected.
    try compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );
}
