"""Tests for the Julia bindings for the TerseTS library."""

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

module TerseTSTests

using Test

include("TerseTS.jl")

@testset "Compress and Decompress Zero Error Small Array" begin
    uncompressed_values = [1.0, 1.0, 1.0, 1.0, 1.0]

    compressed_values = TerseTS.compress(
        uncompressed_values,
        TerseTS.PoorMansCompressionMidrange,
        "{\"abs_error_bound\": 0.0}",
    )
    decompressed_values = TerseTS.decompress(compressed_values)

    @test uncompressed_values == decompressed_values
end

@testset "Compress and Decompress Zero Error Bigger Array" begin
    n = 1000
    uncompressed_values = [sin(0.001 * i) + 0.1 * cos(0.01 * i) for i in 1:n]

    compressed_values = TerseTS.compress(
        uncompressed_values,
        TerseTS.PoorMansCompressionMidrange,
        "{\"abs_error_bound\": 0.0}",
    )
    decompressed_values = TerseTS.decompress(compressed_values)

    @test uncompressed_values == decompressed_values
end

@testset "Compress and Decompress Across Multiple Methods with Zero Error Bound" begin
    methods_to_try = [
        TerseTS.PoorMansCompressionMean,
        TerseTS.PoorMansCompressionMidrange,
        TerseTS.SwingFilter,
        TerseTS.ABCLinearApproximation,
        TerseTS.SwingFilterDisconnected,
        TerseTS.SlideFilter,
    ]

    n = 1000
    uncompressed_values = randn(n) .* 1e3 .+ 0.123

    for method in methods_to_try
        @testset "method = $(method)" begin
            compressed_values = TerseTS.compress(
                uncompressed_values,
                method,
                "{\"abs_error_bound\": 0.0}",
            )
            decompressed_values = TerseTS.decompress(compressed_values)

            @test length(decompressed_values) == length(uncompressed_values)
            @test decompressed_values == uncompressed_values
        end
    end
end


end
