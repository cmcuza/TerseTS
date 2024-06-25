<h1 align="center">
  <img src="docs/tersets.jpg" alt="TerseTS">
</h1>

TerseTS is a library that provides methods for lossless and lossy compressing time series. To match existing literature the methods are organized based on [Time Series Compression Survey](https://dl.acm.org/doi/10.1145/3560814). The library is implemented in Zig and provides a C-API with [bindings](#usage) for other languages.

# Getting Started 
## Compilation
TerseTS can be compiled and cross-compiled from source:
1. Install the latest version of [Zig](https://ziglang.org/)
2. Build TerseTS for development in `Debug` mode using Zig, e.g.,:
   - Linux: `zig build -Dtarget=x86_64-linux`
   - macOS: `zig build -Dtarget=aarch64-macos`
   - Microsoft Windows: `zig build -Dtarget=x86_64-windows`
3. Build TerseTS for deployment in `ReleaseFast`, `ReleaseSafe`, and `ReleaseSmall` mode using Zig, e.g.,:
   - Linux: `zig build -Dtarget=x86_64-linux -Doptimize=ReleaseFast`
   - macOS: `zig build -Dtarget=aarch64-macos -Doptimize=ReleaseFast`
   - Microsoft Windows: `zig build -Dtarget=x86_64-windows -Doptimize=ReleaseFast`

## Usage
TerseTS provides a C-API that is designed to be simple to wrap. Currently, TerseTS includes APIs for the following programming languages which can be used without installation of any dependencies:
<a id="zig-usage-example"></a>
<details>
<summary><strong>Zig Usage Example</strong></summary>

```c
const std = @import("std");
const tersets = @import("path/to/tersets.zig");

pub fn main() void {
   var uncompressed_values = [_]f64{1.0, 2.0, 3.0, 4.0, 5.0};
   std.debug.print("Uncompressed data length: {any}\n", .{uncompressed_values.len});
    
   const config = tersets.Configuration{
        .method = .SwingFilter,
        .error_bound = 0.1,
   };
    
   var compressed_values = try tersets.compress(data[0..], config);
   defer std.heap.page_allocator.free(compressed);

   std.debug.print("Compression successful. Compressed data length: {any}\n", .{compressed_values.items.len});
    
   var decompressed_values = try tersets.decompress(compressed, config);
   defer std.heap.page_allocator.free(decompressed);

   std.debug.print("Decompression successful. Decompressed data length {any}\n", .{decompressed_values.items.len});
}
```

TerseTS provides `./src/tersets.zig` as the single access point and two main functions `compress` and `decompress`. 

For compression, you can select the compression method through the `Configuration` structure with two parameters: the compression method, e.g., `.method=.SwingFilter`, and the error bound, e.g., `.error_bound = 0.1`. 

For decompression, the `Configuration` is not needed as the method is encoded in the compressed values.
</details>

<a id="c-usage-example"></a>
<details>
<summary><strong>C/C++ Usage Example</strong></summary>

```c 
#include "tersets.h"
#include <stdio.h>

int main() {
   double uncompressed_values[] = {1.0, 2.0, 3.0, 4.0, 5.0};
   struct UncompressedValues uncompressed_values = {data, 5};
   
   printf("Uncompressed data length: %lu\n", uncompressed_values.len);
   
   // Configuration for compression.
   // Method 2 is SwingFilter and 0.1 error bound.
   struct Configuration config = {2, 0.0}; 

   // Prepare for compressed data.
   struct CompressedValues compressed_values;
    
   // Compress the data.
   int32_t result = compress(uncompressed_values, &compressed_values, config);
   if (result != 0) {
      printf("Compression failed with error code %d\n", result);
      return -1;
   }

   printf("Compression successful. Decompressed data length: %lu\n", compressed_values.len);
    
   // Prepare for decompressed data.
   struct UncompressedValues decompressed_values;
   
   // Decompress the data.
   int32_t result = decompress(compressed_values, &decompressed_values);
   if (result != 0) {
      printf("Decompression failed with error code %d\n", result);
      return -1;
   }

   printf("Decompression successful. Decompressed data length: %lu\n", decompressed_values.len);
    
   // Free the compressed and decompressed value.
   free(decompressed_values.data);
   free(compressed_values.data);
   return 0;
}
```

TerseTS provides `./bindings/c/tersets.h` as API for C/C++ which should be included in the source code, i.e., `#include "tersets.h"`. You need to [link](#linking) the TerseTS library to the project. 

Ensure that the method field in `Configuration` is set to a valid compression/decompression method supported by `TerseTS`.

Free dynamically allocated memory appropriately to avoid memory leaks.
</details>

<a id="python-usage-example"></a>
<details>
<summary><strong>Python Usage Example</strong></summary>

```python
import random
import sys
from tersets import compress, decompress, Method

uncompressed_values = [1.0, 2.0, 3.0, 4.0, 5.0]
error_bound = 0.1

print("Uncompressed data length: ", len(uncompressed_values))

# Compress using the SwingFilter method.
compressed_values = compress(uncompressed, 0.1, Method.SwingFilter)

print("Compression successful. Compressed data length: ", len(compressed_values))

# Decompress the data.
decompressed_values = decompress(compressed_values)

print("Decompression successful. Decompressed data length: ", len(decompressed_values))
```

TerseTS provides `./bindings/python/tersets/__init__.py` as binding for Python which allows directly importing tersets, i.e., `import tersets`.

The Python binding provides the `Method` enum to provide direct access to the available methods supported by `TerseTS`.
</details>

## Linking:
- **Microsoft Windows**: Link the `tersets.dll` to the project. It can be found in the output folder after compiling TerseTS, by default: `zig-out/lib/tersets.dll`.
- **Linux**: Link the `tersets.so` to the project. It can be found in the output folder after compiling TerseTS, by default: `zig-out/lib/tersets.so`.
- **macOS**: Link the `tersets.dylib` to the project. It can be found in the output folder after compiling TerseTS, by default: `zig-out/lib/tersets.dylib`.

# License
TerseTS is licensed under version 2.0 of the Apache License and a copy of the license is bundled with the program.