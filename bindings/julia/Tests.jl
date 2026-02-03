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

@testset "Compress and Decompress Zero Error" begin


    uncompressed_values = [1.0, 1.0, 1.0, 1.0, 1.0]

    compressed_values = TerseTS.compress(
        uncompressed_values,
        TerseTS.PoorMansCompressionMidrange,
        "{\" absolute_error_bound: 0.0 \"}",
    )
    decompressed_values = TerseTS.decompress(compressed_values)

    @test uncompressed_values == decompressed_values
end

end
