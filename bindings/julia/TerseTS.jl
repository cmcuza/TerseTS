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

"""
 Mirror TerseTS Method Enum.
"""
@enum Method::UInt8 begin
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
Locate the TerseTS library for this system.
"""
function findlibrary()::String
    repository_root = joinpath(@__DIR__, "..", "..")

    if Sys.iswindows()
        # Zig uses zig-out/bin for dynamic linked libraries on Windows.
        joinpath(repository_root, "zig-out", "bin", "tersets.dll")
    elseif Sys.isapple()
        joinpath(repository_root, "zig-out", "lib", "libtersets.dylib")
    elseif Sys.isunix()
        joinpath(repository_root, "zig-out", "lib", "libtersets.so")
    else
        error("Only FreeBSD, Linux, macOS, and Windows is supported.")
    end
end

"""
Compress an `AbstractVector` of `Float64` values with a TerseTS compression `method` according to `configuration`.
"""
function compress(
    uncompressed_values::AbstractVector{Float64},
    method::Method,
    configuration::AbstractString,
)::AbstractVector{UInt8}
    uncompressed_values_struct =
        UncompressedValues(pointer(uncompressed_values), length(uncompressed_values))
    compressed_values_struct_ref = Ref{CompressedValues}(CompressedValues(C_NULL, 0))

    tersets_error = @ccall findlibrary().compress(
        uncompressed_values_struct::UncompressedValues,
        compressed_values_struct_ref::Ref{CompressedValues},
        method::UInt8,
        configuration::Cstring,
    )::Cint

    if tersets_error != 0
        error("compress failed: $tersets_error")
    end

    compressed_values_struct = compressed_values_struct_ref[]

    compressed_values_view = unsafe_wrap(
        Vector{UInt8},
        compressed_values_struct.data,
        compressed_values_struct.len,
    )
    compressed_values = copy(compressed_values_view)

    @ccall findlibrary().freeCompressedValues(
        compressed_values_struct_ref::Ref{CompressedValues},
    )::Cvoid

    compressed_values
end

"""
Decompress a TerseTS-compressed `AbstractVector` into an `AbstractVector` of `Float64`.
"""
function decompress(compressed_values::AbstractVector{UInt8})::AbstractVector{Float64}
    compressed_values_struct =
        CompressedValues(pointer(compressed_values), length(compressed_values))
    uncompressed_values_struct_ref = Ref{UncompressedValues}(UncompressedValues(C_NULL, 0))

    tersets_error = @ccall findlibrary().decompress(
        compressed_values_struct::CompressedValues,
        uncompressed_values_struct_ref::Ref{UncompressedValues},
    )::Cint

    if tersets_error != 0
        error("decompress failed: $tersets_error")
    end

    uncompressed_values_struct = uncompressed_values_struct_ref[]

    uncompressed_values_view = unsafe_wrap(
        Vector{Float64},
        uncompressed_values_struct.data,
        uncompressed_values_struct.len,
    )
    uncompressed_values = copy(uncompressed_values_view)

    @ccall findlibrary().freeUncompressedValues(
        uncompressed_values_struct_ref::Ref{UncompressedValues},
    )::Cvoid

    uncompressed_values
end

end
