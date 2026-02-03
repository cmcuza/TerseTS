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
Mirror TerseTS `Method` enum.
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
    data::Ref{Float64}
    len::Csize_t
end

"""
A pointer to compressed values and the number of bytes.
"""
struct CompressedValues
    data::Ref{UInt8}
    len::Csize_t
end

"""
Locate the TerseTS library for this operation.
"""
function findlibrary()
    if Sys.isapple()
        println("Hello Mac")
    elseif Sys.isunix()
        println("Hello Unix")
    elseif Sys.iswindows()
        println("Hello Windowos")
        library_name = "tersets.dll"
    else
        error("Could not find TerseTS: looked '*{library_name}' in {library_folder}")
    end

    "/home/kejser/Projects/2023-TerseTS/TerseTS/zig-out/lib/libtersets.so"
end

"""
Compress a `Vector` of `Float64` values with a selected TerseTS `method` according to `configuration`.
"""
function compress(
    uncompressed_values::AbstractVector{Float64},
    method::Method,
    configuration::AbstractString,
)
    AbstractVector{UInt8}
    uncompressed_values_ref = convert(Ref{Float64}, uncompressed_values)
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
