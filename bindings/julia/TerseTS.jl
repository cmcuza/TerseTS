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

@enum Method begin
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

struct UncompressedValues
    data::Ptr{Float64}
    len::Csize_t
end

struct CompressedValues
    data::Ptr{UInt8}
    len::Csize_t
end

function find_library()
    if  Sys.isapple()
        println("Hello Mac")
    elseif Sys.isunix()
        println("Hello Unix")
    elseif Sys.iswindows()
        println("Hello Windowos")
        library_name = "tersets.dll"
    else
        error("Could not find TerseTS: looked '*{library_name}' in {library_folder}")
    end

    "/home/kejser/Projects/2023-TerseTS/TerseTS/zig-out/lib/"
end

function compress()
end

function decompress()
end

function freeCompressedValues()
end

function freeUncompressedValues()
end

function __init__()
    find_library()
end

end
