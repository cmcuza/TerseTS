"""Julia bindings for the TerseTS library."""

# Copyright 2026 TerseTS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

module TerseTS

using Libdl

# TODO: Fix segmentation and garbage collection errors.
# https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/
# https://docs.julialang.org/en/v1/base/c/

# TODO: Make all code and tests use Julia's style guide.
# https://docs.julialang.org/en/v1/manual/style-guide/

# TODO: Format all code and tests using JuliaFormatter.jl.
# https://domluna.github.io/JuliaFormatter.jl/stable/

"""
Mirror TerseTS `Method` Enum.
"""
@enum Method begin
    PoorMansCompressionMidrange = 0
    PoorMansCompressionMean = 1
    SwingFilter = 2
    SwingFilterDisconnected = 3
    SlideFilter = 4
    SimPiece = 5
    PiecewiseConstantHistogram = 6
    PiecewiseLinearHistogram = 7
    ABCLinearApproximation = 8
    VisvalingamWhyatt = 9
    SlidingWindow = 10
    BottomUp = 11
    MixPiece = 12
    BitPackedQuantization = 13
    RunLengthEncoding = 14
    NonLinearApproximation = 15
    SerfQT = 16
end

"""
A pointer to uncompressed values and the number of values.
"""
struct UncompressedValues
    data::Ptr{Float64}
    len::Csize_t
end

"""
A pointer to compressed values and the number of bytes.
"""
struct CompressedValues
    data::Ptr{UInt8}
    len::Csize_t
end

"""
Locate the TerseTS library for this operation.
"""
function findlibrary()
    library_name = if Sys.isapple()
        "libtersets.dylib"
    elseif Sys.isunix()
        "libtersets.so"
    elseif Sys.iswindows()
        "tersets.dll"
    else
        error("Only FreeBSD, Linux, macOS, and Windows is supported.")
    end

    return normpath(joinpath(@__DIR__, "..", "..", "zig-out", "lib", library_name))
end

function __init__()
    _lib[] = Libdl.dlopen(findlibrary())
end

"""
Compress a `Vector` of `Float64` values with a selected TerseTS `method` according to `configuration`.
"""
function compress(
    uncompressed_values::AbstractVector{Float64},
    method::Method,
    configuration::AbstractString,
)   
    # Ensure contiguous Float64 storage
    xvec = x isa Vector{Float64} ? x : collect(Float64, x)
    AbstractVector{UInt8} uncompressed_values_ref = convert(Ptr{Float64}, uncompressed_values)
    
    uncompressed_values_struct =
        UncompressedValues(uncompressed_values_ref, length(uncompressed_values))
    
    compressed_values_struct = CompressedValues(C_NULL, 0)

    tersets_error = @ccall findlibrary().compress(
        uncompressed_values_struct::Ref{UncompressedValues},
        compressed_values_struct::Ref{CompressedValues},
        method::Method,
        configuration::Cstring,
    )::Cint

    if tersets_error != 0
        throw("compress failed: $tersets_error")
    end

    compressed_values = convert(AbstractVector{UInt8}, uncompressed_values_struct.data)
    @ccall findlibrary().freeCompressedValues(
        compressed_values_struct::Ref{CompressedValues},
    )::Cvoid

    compressed_values
end

"""
Decompress a TerseTS-compressed `Vector` into a `Vector` of `Float64`.
"""
function decompress(compressed_values::AbstractVector{UInt8})
    AbstractVector{Float64}
    compressed_values_ref = convert(Ref{UInt8}, compressed_values)
    compressed_values_struct =
        CompressedValues(compressed_values_ref, length(compressed_values))
    uncompressed_values_struct = UncompressedValues(C_NULL, 0)

    tersets_error = @ccall findlibrary().decompress(
        compressed_values_struct::Ref{CompressedValues},
        uncompressed_values_struct::Ref{UncompressedValues},
    )::Cint

    if tersets_error != 0
        throw("decompress failed: $tersets_error")
    end

    uncompressed_values = convert(AbstractVector{Float64}, compressed_values_struct.data)
    @ccall findlibrary().freeUncompressedValues(
        compressed_values_struct::Ref{UncompressedValues},
    )::Cvoid

    uncompressed_values
end

end
