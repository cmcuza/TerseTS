# ![Alt text](docs/tersetspng.jpg "TerseTS")

# TerseTS
TerseTS is a library that provides methods for lossless and lossy compressing time series. To match existing litterature the methods are organized based on [Time Series Compression Survey](https://dl.acm.org/doi/10.1145/3560814). The library is implemented in Zig and provides a C-API with [bindings](#Installation) for other languages.

# Installation
TerseTS can be compiled and cross-compiled from source:
1. Install the latest version of [Zig](https://ziglang.org/)
2. Build TerseTS for development in `Debug` mode using Zig, e.g.,:
   - Linux: `zig build -Dtarget=x86_64-linux`
   - macOS: `zig build -Dtarget=aarch64-macos`
   - Microsoft Windows: `zig build -Dtarget=x86_64-windows`
3. Build TerseTS for deployment in `ReleaseFast`, `ReleaseSafe`, and `ReleaseSmall` mode using Zig, e.g.,:
   - Linux: `zig build -Dtarget=x86_64-linux`
   - macOS: `zig build -Dtarget=aarch64-macos`
   - Microsoft Windows: `zig build -Dtarget=x86_64-windows`

# Usage
TerseTS's API is inspired by [libpressio](https://github.com/robertu94/libpressio) and currently provides these methods:
```zig
export fn compress(input: Input, output: *Output) i32;
```

The data to be compressed and the compression method to use is provided as an instance of `Input`, the compressed data is written to an instance of `Output`, and if an error occurs a non-zero integer is returned. Thus, only the instance of `Input` has to be changed to use a completely compression method or change the configuration of a compression method.


# Bindings
TerseTS provides a C-API that is designed to be simple to wrap. Currently, TerseTS includes bindings for the current languages which can be used without installation of any dependencies:
- [Zig](tersets/capi.zig)
- [C](tersets/capi.zig)
- [C++](tersets/capi.zig) 
- [Python](bindings/tersets.py) using [ctypes](https://docs.python.org/3/library/ctypes.html)

# Links
Information used to implement the libary, create the bindings, and write the README:
- https://ziglearn.org/
- https://ziglang.org/documentation/master/
- https://dev.to/tonetheman/call-zig-from-python-3fmo
- https://gitlab.com/nicolalandro/study-zig-python-lib
- https://github.com/kristoff-it/zig-cuckoofilter
- https://github.com/zigimg/zigimg
- https://github.com/adamserafini/zaml
- https://github.com/robertu94/libpressio
- https://github.com/cmcuza/tersets

# License
TerseTS is licensed under version 2.0 of the Apache License and a copy of the license is bundled with the program.