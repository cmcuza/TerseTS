const std = @import("std");
const ArrayList = std.ArrayList;
const print = std.debug.print;

// Import the mix_piece module - adjust path as needed
const mix_piece = @import("./functional/mix_piece.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Example 1: Simple linear data with some noise
    print("=== Example 1: Linear data with noise ===\n", .{});
    var linear_data = ArrayList(f64).init(allocator);
    defer linear_data.deinit();

    // Generate y = 2x + 1 with some noise
    for (0..20) |i| {
        const x: f64 = @as(f64, @floatFromInt(i));
        const noise = (@as(f64, @floatFromInt(i % 3)) - 1.0) * 0.05; // Small noise
        try linear_data.append(2.0 * x + 1.0 + noise);
    }

    try testCompressionDecompression(allocator, linear_data.items, 0.1, "Linear with noise");

    // Example 2: Sine wave (should be harder to compress)
    print("\n=== Example 2: Sine wave ===\n", .{});
    var sine_data = ArrayList(f64).init(allocator);
    defer sine_data.deinit();

    for (0..50) |i| {
        const x: f64 = @as(f64, @floatFromInt(i)) * 0.2;
        try sine_data.append(@sin(x));
    }

    try testCompressionDecompression(allocator, sine_data.items, 0.1, "Sine wave");

    // Example 3: Step function (should compress very well)
    print("\n=== Example 3: Step function ===\n", .{});
    var step_data = ArrayList(f64).init(allocator);
    defer step_data.deinit();

    for (0..100) |i| {
        const value: f64 = if (i < 25) 1.0 else if (i < 50) 2.0 else if (i < 75) 1.5 else 3.0;
        try step_data.append(value);
    }

    try testCompressionDecompression(allocator, step_data.items, 0.05, "Step function");

    // Example 4: Data with trend (Mix-Piece should handle this better than Sim-Piece)
    print("\n=== Example 4: Data with trend ===\n", .{});
    var trend_data = ArrayList(f64).init(allocator);
    defer trend_data.deinit();

    for (0..50) |i| {
        const x: f64 = @as(f64, @floatFromInt(i));
        const trend = x * 0.5;
        const seasonal = 3.0 * @sin(x * 0.3);
        const noise = (@as(f64, @floatFromInt(i % 5)) - 2.0) * 0.1;
        try trend_data.append(trend + seasonal + noise);
    }

    try testCompressionDecompression(allocator, trend_data.items, 0.2, "Data with trend");

    // Example 5: Piecewise data with different slopes
    print("\n=== Example 5: Piecewise linear segments ===\n", .{});
    var piecewise_data = ArrayList(f64).init(allocator);
    defer piecewise_data.deinit();

    // Segment 1: slope = 1
    for (0..10) |i| {
        try piecewise_data.append(@as(f64, @floatFromInt(i)));
    }
    // Segment 2: slope = -0.5
    for (10..20) |i| {
        try piecewise_data.append(10.0 - 0.5 * @as(f64, @floatFromInt(i - 10)));
    }
    // Segment 3: slope = 2
    for (20..30) |i| {
        try piecewise_data.append(5.0 + 2.0 * @as(f64, @floatFromInt(i - 20)));
    }

    try testCompressionDecompression(allocator, piecewise_data.items, 0.1, "Piecewise linear");

    // Example 6: Constant segments (should group well)
    print("\n=== Example 6: Constant segments ===\n", .{});
    var constant_data = ArrayList(f64).init(allocator);
    defer constant_data.deinit();

    // Multiple segments with the same value
    for (0..5) |_| try constant_data.append(1.0);
    for (0..5) |_| try constant_data.append(2.0);
    for (0..5) |_| try constant_data.append(1.0); // Same as first segment
    for (0..5) |_| try constant_data.append(3.0);
    for (0..5) |_| try constant_data.append(2.0); // Same as second segment

    try testCompressionDecompression(allocator, constant_data.items, 0.01, "Constant segments");

    // Example 7: Small dataset
    print("\n=== Example 7: Small dataset ===\n", .{});
    var small_data = ArrayList(f64).init(allocator);
    defer small_data.deinit();
    const small_values = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try small_data.appendSlice(&small_values);

    try testCompressionDecompression(allocator, small_data.items, 0.1, "Small dataset");
}

fn testCompressionDecompression(
    allocator: std.mem.Allocator,
    data: []const f64,
    error_bound: f32,
    description: []const u8,
) !void {
    print("Testing: {s}\n", .{description});
    print("Original data ({} points): ", .{data.len});
    for (data[0..@min(10, data.len)]) |val| {
        print("{d:.3} ", .{val});
    }
    if (data.len > 10) print("...", .{});
    print("\n", .{});

    // Compress using Mix-Piece
    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    try mix_piece.compressMixPiece(data, &compressed, allocator, error_bound);

    print("Compressed size: {} bytes\n", .{compressed.items.len});
    print("Compression ratio: {d:.2}:1\n", .{
        @as(f64, @floatFromInt(data.len * 8)) / @as(f64, @floatFromInt(compressed.items.len)),
    });

    // Parse header to show structure breakdown
    if (compressed.items.len >= 3 * @sizeOf(usize)) {
        const header = std.mem.bytesAsSlice(usize, compressed.items[0 .. 3 * @sizeOf(usize)]);
        print("  - Part 1 (same intercept groups): {} groups\n", .{header[0]});
        print("  - Part 2 (cross intercept groups): {} groups\n", .{header[1]});
        print("  - Part 3 (ungrouped segments): {} segments\n", .{header[2]});
    }

    // Decompress
    var decompressed = ArrayList(f64).init(allocator);
    defer decompressed.deinit();

    try mix_piece.decompressMixPiece(compressed.items, &decompressed, allocator);

    print("Decompressed data: ", .{});
    for (decompressed.items[0..@min(10, decompressed.items.len)]) |val| {
        print("{d:.3} ", .{val});
    }
    if (decompressed.items.len > 10) print("...", .{});
    print("\n", .{});

    // Verify length
    if (data.len != decompressed.items.len) {
        print("ERROR: Length mismatch! Original: {}, Decompressed: {}\n", .{ data.len, decompressed.items.len });
        return;
    }

    // Check error bounds
    var max_error: f64 = 0.0;
    var avg_error: f64 = 0.0;
    var error_positions = ArrayList(usize).init(allocator);
    defer error_positions.deinit();

    for (data, decompressed.items, 0..) |orig, decomp, i| {
        const diff = @abs(orig - decomp);
        max_error = @max(max_error, diff);
        avg_error += diff;

        if (diff > error_bound) {
            try error_positions.append(i);
        }
    }
    avg_error /= @as(f64, @floatFromInt(data.len));

    print("Max error: {d:.6} (bound: {d:.3})\n", .{ max_error, error_bound });
    print("Avg error: {d:.6}\n", .{avg_error});
    print("Within bounds: {}\n", .{max_error <= error_bound});

    if (error_positions.items.len > 0) {
        print("ERROR: Found {} positions exceeding error bound:\n", .{error_positions.items.len});
        for (error_positions.items[0..@min(5, error_positions.items.len)]) |pos| {
            print("  Position {}: orig={d:.6}, decomp={d:.6}, error={d:.6}\n", .{
                pos,
                data[pos],
                decompressed.items[pos],
                @abs(data[pos] - decompressed.items[pos]),
            });
        }
        if (error_positions.items.len > 5) print("  ... and {} more\n", .{error_positions.items.len - 5});
    }

    print("Status: {s}\n", .{if (max_error <= error_bound) "PASS ✓" else "FAIL ✗"});
}
