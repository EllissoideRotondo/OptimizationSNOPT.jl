@testset "Solutions preserve SciML parameter access" begin
    opt = SnoptOptimizer()
    f = OptimizationFunction((x, p) -> sum(abs2, x), AutoForwardDiff())
    cache = init(OptimizationProblem(f, [1.0], [2.0]), opt)
    sol = SciMLBase.build_solution(cache, opt, [0.0], 0.0)
    @test sol.ps[1] == 2.0
    @test sol.u == [0.0]
    @test sol.trace === nothing
end

@testset "Adapter workspace operations share the SNOPT process lock" begin
    excluded = lock(SNOPT.SNOPT_LOCK) do
        fetch(@async begin
            acquired = trylock(OptimizationSNOPT.SNOPT_GLOBAL_LOCK)
            acquired && unlock(OptimizationSNOPT.SNOPT_GLOBAL_LOCK)
            !acquired
        end)
    end
    @test excluded
end

@testset "Additional option names ignore case and repeated whitespace" begin
    for key in ("MAJOR ITERATIONS LIMIT", "Major Iterations Limit", "major   iterations limit")
        @test_throws ArgumentError SnoptOptimizer(additional_options = Dict(key => 2))
    end
    @test_throws ArgumentError SnoptOptimizer(additional_options = Dict(
        "Linesearch tolerance" => 0.9, "LINESEARCH TOLERANCE" => 0.8))
    opt = SnoptOptimizer(additional_options = Dict("LINESEARCH   TOLERANCE" => 0.9))
    @test opt.additional_options == Dict("Linesearch tolerance" => 0.9)
end
