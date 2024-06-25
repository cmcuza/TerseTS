<h1 align="center">
  <img src="docs/tersets.jpg" alt="TerseTS">
</h1>

TerseTS is a library that provides methods for lossless and lossy compressing time series. To match existing literature the methods are organized based on [Time Series Compression Survey](https://dl.acm.org/doi/10.1145/3560814). The library is implemented in Zig and provides a C-API with [bindings](#Installation) for other languages.

# Usage
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
4. Using TerseTS in Different Languages:
   - TerseTS provides a C-API that is designed to be simple to wrap. Currently, TerseTS includes bindings for the following programming languages which can be used without installation of any dependencies.
      - [Zig](src/tersets.zig). Check an example in [Zig Usage Example](#zig-usage-example).
      - [C](bindings/c/tersets.h). Check an example in [C/C++ Usage Example](#c-usage-example).
      - [C++](bindings/c/tersets.h). Check an example in [C/C++ Usage Example](#c-usage-example).
      - [Python](bindings/python/tersets) using [ctypes](https://docs.python.org/3/library/ctypes.html). Check an example in [Python Usage Example](#python-usage-example).
<details>
<summary><strong>Zig</strong></summary>

### Zig Usage Example


```rust
const std = @import("std");
const tersets = @import("path/to/tersets.zig");

pub fn main() void {
    var data = [_]f64{1.0, 2.0, 3.0, 4.0, 5.0};
    const config = tersets.Configuration{
        .method = .SwingFilter,
        .error_bound = 0.0,
    };
    
    var compressed = try tersets.compress(data[0..], config);
    defer std.heap.page_allocator.free(compressed);

    var decompressed = try tersets.decompress(compressed, config);
    defer std.heap.page_allocator.free(decompressed);

    std.debug.print("Decompression successful: {any}\n", .{decompressed});
}
```
</details>

<details>
<summary><strong>C</strong></summary>

### C Usage Example

```c 
#include "tersets.h"
#include <stdio.h>

int main() {
    // Example uncompressed data
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    struct UncompressedValues uncompressed_values = {data, 5};

    // Configuration for compression
    struct Configuration config = {2, 0.0}; // Method 2 (e.g., SwingFilter), 0.0 error bound

    // Prepare for compressed data
    struct CompressedValues compressed_values;
    
    // Compress the data
    int32_t result = compress(uncompressed_values, &compressed_values, config);
    if (result != 0) {
        printf("Compression failed with error code %d\n", result);
        return -1;
    }

    printf("Compression successful. Compressed data length: %lu\n", compressed_values.len);
    
    // Prepare for decompressed data
    struct UncompressedValues decompressed_values;
   
    // Decompress the data
    int32_t result = decompress(compressed_values, &decompressed_values);
    if (result != 0) {
        printf("Decompression failed with error code %d\n", result);
        return -1;
    }

    printf("Decompression successful. Uncompressed data length: %lu\n", decompressed_values.len);
    
    // Free the uncompressed data if dynamically allocated (not shown here)
    // free(decompressed_values.data);
    // free(compressed_values.data);
    return 0;
}
```
### Notes
1. Include the Header and Link the Library.
2. Add #include "tersets.h" in your source files and link the TerseTS library to your project.
3. Ensure that the method field in `Configuration` is set to a valid compression/decompression method supported by `TerseTS`.
4. Free dynamically allocated memory appropriately to avoid memory leaks.
</details>

<details>
<summary><strong>Python</strong></summary>

### Python Usage Example

```python
import random
import sys
from tersets import compress, decompress, Method

# Number of values to generate for each test.
TEST_VALUE_COUNT = 1000

# Example Usage

# Generate some random uncompressed data
uncompressed = [random.uniform() for _ in range(TEST_VALUE_COUNT)]

# Compress the data with zero error using the SwingFilter method
compressed = compress(uncompressed, 0.0, Method.SwingFilter)

# Decompress the data back to its original form
decompressed = decompress(compressed)

# Verify that the decompressed data matches the original data
assert uncompressed == decompressed
print("Compression and decompression were successful.")
```
### Notes
   1. Modify the `__init__.py` file in the TerseTS Python bindings to link the correct shared library for your operating system. Specifically, change the `library_path` variable to reflect the path to TerseTS's library.
   2. Import the necessary functions and classes from the TerseTS Python module.

</details>




# License
TerseTS is licensed under version 2.0 of the Apache License and a copy of the license is bundled with the program.