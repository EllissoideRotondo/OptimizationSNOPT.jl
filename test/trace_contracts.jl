using OptimizationSNOPT
using Test

@testset "Trace frequencies reject Boolean values" begin
    # Given a Boolean where the API requires a positive integer,
    # when a trace level is constructed, then construction must fail.
    for constructor in (SnoptTraceMinimal, SnoptTraceAll)
        @test_throws ArgumentError constructor(print_frequency = true)
        @test_throws ArgumentError constructor(store_frequency = false)
        @test_throws ArgumentError constructor(true)
    end
end

@testset "Trace frequencies accept positive Integer values" begin
    for constructor in (SnoptTraceMinimal, SnoptTraceAll)
        for frequency in (Int8(1), Int32(2), Int128(3))
            level = constructor(
                print_frequency = frequency,
                store_frequency = frequency,
            )
            @test level.print_frequency == frequency
            @test level.store_frequency == frequency
        end
    end
end
