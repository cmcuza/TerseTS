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

# --- API enum (matches your C enum) ---
@enum Method::UInt8 begin
    PoorMansCompressionMidrange   = 0
    PoorMansCompressionMean       = 1
    SwingFilter                   = 2
    SwingFilterDisconnected       = 3
    SlideFilter                   = 4
    SimPiece                      = 5
    PiecewiseConstantHistogram    = 6
    PiecewiseLinearHistogram      = 7
    ABCLinearApproximation        = 8
    VisvalingamWhyatt             = 9
    SlidingWindow                 = 10
    BottomUp                      = 11
    MixPiece                      = 12
    BitPackedQuantization         = 13
    RunLengthEncoding             = 14
    NonLinearApproximation        = 15
    SerfQT                        = 16
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
function findlibrary()::String
    if haskey(ENV, "TERSETS_LIB")
        return ENV["TERSETS_LIB"]
    end

    libname = if Sys.iswindows()
        "tersets.dll"
    elseif Sys.isapple()
        "libtersets.dylib"
    else
        "libtersets.so"
    end

    # Zig often uses zig-out/bin for .dll on Windows.
    base = normpath(joinpath(@__DIR__, "..", ".."))
    p_bin = joinpath(base, "zig-out", "bin", libname)
    p_lib = joinpath(base, "zig-out", "lib", libname)

    if isfile(p_bin)
        return p_bin
    elseif isfile(p_lib)
        return p_lib
    else
        # Let the system fail if the file does not exist.
        return p_bin  
    end
end

"""Lazy load the TerseTS library."""
const _LIB = Ref{Union{Nothing,Libdl.LazyLibrary}}(nothing)

"""
Get the lazy-loaded TerseTS library.
"""
@inline function _lib()::Libdl.LazyLibrary
    if _LIB[] === nothing
        _LIB[] = Libdl.LazyLibrary(findlibrary())
    end
    return _LIB[]::Libdl.LazyLibrary
end

"""
Check the returned code from a TerseTS call.
"""
@inline function _check(returned_code::Cint)
    returned_code == 0 && return
    error("TerseTS call failed with error code: $(Int(returned_code))")
end

"""
Compress a `Vector` of `Float64` values with a selected TerseTS `method` according to `configuration`.
"""
function compress(uncompressed_values::AbstractVector{Float64}, method::Method, configuration::AbstractString)::Vector{UInt8}
    uncompressed_vector = uncompressed_values isa Vector{Float64} ? uncompressed_values : collect(Float64, uncompressed_values)
    
    compressed_values_ref = Ref{CompressedValues}(CompressedValues(C_NULL, 0))
    configuration_string = String(configuration)

    GC.@preserve uncompressed_vector configuration_string begin
        uncompressed_values_view = UncompressedValues(pointer(uncompressed_vector), Csize_t(length(uncompressed_vector)))
        returned_code = ccall((:compress, _lib()), Cint,
                   (UncompressedValues, Ref{CompressedValues}, UInt8, Cstring),
                   uncompressed_values_view, compressed_values_ref, UInt8(method), configuration_string)
        _check(returned_code)
    end

    compressed_values = compressed_values_ref[]
    compressed_values.data == C_NULL && error("compress: returned NULL compressed representation.")

    bytes_view = unsafe_wrap(Vector{UInt8}, compressed_values.data, Int(compressed_values.len); own=false)
    result = copy(bytes_view)

    ccall((:freeCompressedValues, _lib()), Cvoid, (Ref{CompressedValues},), compressed_values_ref)
    return result
end

"""
Decompress a TerseTS-compressed `Vector` into a `Vector` of `Float64`.
"""
function decompress(compressed_bytes::AbstractVector{UInt8})::Vector{Float64}
    compressed_vector = compressed_bytes isa Vector{UInt8} ? compressed_bytes : collect(UInt8, compressed_bytes)

    uncompressed_values_ref = Ref{UncompressedValues}(UncompressedValues(C_NULL, 0))

    GC.@preserve compressed_vector begin
        compressed_values_view = CompressedValues(pointer(compressed_vector), Csize_t(length(compressed_vector)))
        returned_code = ccall((:decompress, _lib()), Cint,
                   (CompressedValues, Ref{UncompressedValues}),
                   compressed_values_view, uncompressed_values_ref)
        _check(returned_code)
    end
    
    uncompressed_values = uncompressed_values_ref[]
    uncompressed_values.data == C_NULL && error("decompress: returned NULL decompressed values.")

    vals_view = unsafe_wrap(Vector{Float64}, uncompressed_values.data, Int(uncompressed_values.len); own=false)
    result = copy(vals_view)

    ccall((:freeUncompressedValues, _lib()), Cvoid, (Ref{UncompressedValues},), uncompressed_values_ref)
    return result
end

end # module TerseTS
