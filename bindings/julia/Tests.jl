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
    uncompressed_values = [sin(0.001 * i) + 0.1 * cos(0.01 * i) for i = 1:n]

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
            compressed_values =
                TerseTS.compress(uncompressed_values, method, "{\"abs_error_bound\": 0.0}")
            decompressed_values = TerseTS.decompress(compressed_values)

            @test length(decompressed_values) == length(uncompressed_values)
            @test decompressed_values == uncompressed_values
        end
    end
end

@testset "Julia Method Enum Matches C and Zig Enums" begin
    # Get repository root
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    c_file = joinpath(repo_root, "bindings", "c", "tersets.h")
    zig_file = joinpath(repo_root, "src", "tersets.zig")

    # Helper function to extract enum members from C header
    function extract_c_enum()
        content = read(c_file, String)
        # Match enum Method { ... }
        m = match(r"enum\s+Method\s*\{([^}]*)\}", content)
        if m === nothing
            error("Could not find Method enum in tersets.h")
        end

        body = m.captures[1]
        members = String[]
        values = Int[]
        next_value = 0

        for line in split(body, '\n')
            line = strip(line)
            line = rstrip(line, ',')
            isempty(line) && continue

            if contains(line, '=')
                parts = split(line, '=')
                name = strip(parts[1])
                value = parse(Int, strip(parts[2]))
                push!(members, name)
                push!(values, value)
                next_value = value + 1
            else
                push!(members, line)
                push!(values, next_value)
                next_value += 1
            end
        end

        return members, values
    end

    # Helper function to extract enum members from Zig source
    function extract_zig_enum()
        content = read(zig_file, String)
        # Match pub const Method = enum { ... }
        m = match(r"pub const Method = enum\s*\{([^}]*)\}", content)
        if m === nothing
            error("Could not find Method enum in tersets.zig")
        end

        body = m.captures[1]
        members = String[]

        for line in split(body, '\n')
            line = strip(line)
            line = rstrip(line, ',')
            isempty(line) && continue
            push!(members, line)
        end

        return members
    end

    # Extract enums from source files
    c_members, c_values = extract_c_enum()
    zig_members = extract_zig_enum()

    # Get Julia enum members
    julia_members = String[string(m) for m in instances(TerseTS.Method)]
    julia_values = Int[Int(m) for m in instances(TerseTS.Method)]

    # Test that all enums match
    @test zig_members == c_members
    @test zig_members == julia_members

    # Test that C enum values are sequential starting from 0
    @test c_values == collect(0:(length(c_members)-1))

    # Test that Julia enum values match C enum values
    @test julia_values == c_values
end
end
