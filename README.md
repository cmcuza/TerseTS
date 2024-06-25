<h1 align="center">
  <img src="docs/tersets.jpg" alt="TerseTS">
</h1>

TerseTS is a library that provides methods for lossless and lossy compressing time series. To match existing literature the methods are organized based on [Time Series Compression Survey](https://dl.acm.org/doi/10.1145/3560814). The library is implemented in Zig and provides a C-API with [bindings](#usage) for other languages.

# Using TerseTS

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

## Getting Started
TerseTS provides a C-API that is designed to be simple to wrap. Currently, TerseTS includes bindings for the following programming languages which can be used without installation of any dependencies:
<a id="zig-usage-example"></a>
<details>
<summary><strong>Zig Usage Example</strong></summary>

```c
const std = @import("std");
const tersets = @import("path/to/tersets.zig");

pub fn main() void {
    var uncompressed_values = [_]f64{1.0, 2.0, 3.0, 4.0, 5.0};
    const config = tersets.Configuration{
        .method = .SwingFilter,
        .error_bound = 0.1,
    };
    
    var compressed_values = try tersets.compress(data[0..], config);
    defer std.heap.page_allocator.free(compressed);

    var decompressed_values = try tersets.decompress(compressed, config);
    defer std.heap.page_allocator.free(decompressed);

    std.debug.print("Decompression successful: {any}\n", .{decompressed_values.len});
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

    // Configuration for compression.
    struct Configuration config = {2, 0.0}; // Method 2 (e.g., SwingFilter), and 0.1 error bound.

    // Prepare for compressed data.
    struct CompressedValues compressed_values;
    
    // Compress the data.
    int32_t result = compress(uncompressed_values, &compressed_values, config);
    if (result != 0) {
        printf("Compression failed with error code %d\n", result);
        return -1;
    }

    printf("Compression successful. Compressed data length: %lu\n", compressed_values.len);
    
    // Prepare for decompressed data.
    struct UncompressedValues decompressed_values;
   
    // Decompress the data.
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

TerseTS provides `./bindings/c/tersets.h` as binding for C/C++, thus, the `#include "tersets.h"` in the source code. You need to link the TerseTS library to your project [linking](#linking).

-  Ensure that the method field in `Configuration` is set to a valid compression/decompression method supported by `TerseTS`.

-  Free dynamically allocated memory appropriately to avoid memory leaks.
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

# Compress using the SwingFilter method.
compressed_values = compress(uncompressed, 0.1, Method.SwingFilter)

# Decompress the data
decompressed = decompress(compressed_values)

print("Decompression successful. Uncompressed data length: ", len(decompressed))
```
To use TerseTS from Python first ensure the `__init__.py` file in `binding/python/__init__.py` links the correct shared library. Specifically, change the `library_path` variable to reflect the path to TerseTS's library [linking](#linking). For example, in Windows: `library_path = path/to/tersets/zig-out/lib/tersets.dll`.

</details>

## Linking:

* **For Windows:** Link the `tersets.dll` to your project. `tersets.dll` can be found in the output folder after compiling TersetTS, by default: `zig-out/lib/tersets.dll`.  

* **For Linux:** Link the `tersets.so` to your project. `tersets.so` can be found in the output folder after compiling TersetTS, by default: `zig-out/lib/tersets.so`.





# License
TerseTS is licensed under version 2.0 of the Apache License and a copy of the license is bundled with the program.