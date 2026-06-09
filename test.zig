const std = @import("std");
const tersets = @import("src/tersets.zig");
var gpa = std.heap.DebugAllocator(.{}){};
const allocator = gpa.allocator();

pub fn main() !void {
   const uncompressed_values = [_]f64{1.0, 2.0, 3.0, 4.0, 5.0};
   std.debug.print("Uncompressed data length: {any}\n", .{uncompressed_values.len});

   // Configuration for compression.
   // The supported compression methods are specified in tersets.zig.
   const method = tersets.Method.SwingFilter;
   // The supported configurations are specified in configuration.zig.
   const configuration = "{ \"abs_error_bound\": 0.1 }";

   // Compress the data.
   var compressed_values = try tersets.compress(allocator, &uncompressed_values, method, configuration);
   // The compressed values point to dynamically allocated data that should be deallocated.
   defer compressed_values.deinit(allocator);

   std.debug.print("Compression successful. Compressed data length: {any}\n", .{compressed_values.items.len});

   // Decompress the data.
   var decompressed_values = try tersets.decompress(allocator, compressed_values.items);
   // The decompressed values point to dynamically allocated data that should be deallocated.
   defer decompressed_values.deinit(allocator);

   std.debug.print("Decompression successful. Decompressed data length {any}\n", .{decompressed_values.items.len});
}
