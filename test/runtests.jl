using OptimizationBase, OptimizationSnopt
using Snopt
using Zygote
using Symbolics
using Test
using SparseArrays
using ModelingToolkit
using ReverseDiff

@testset "SnoptOptimizer validation" begin
    @test_throws ArgumentError SnoptOptimizer(major_print_level = -1)
    @test_throws ArgumentError SnoptOptimizer(minor_print_level = -1)
    @test_throws ArgumentError SnoptOptimizer(major_iterations_limit = 0)
    @test_throws ArgumentError SnoptOptimizer(major_iterations_limit = -10)
    @test_throws ArgumentError SnoptOptimizer(minor_iterations_limit = 0)
    @test_throws ArgumentError SnoptOptimizer(major_optimality_tolerance = 0.0)
    @test_throws ArgumentError SnoptOptimizer(major_optimality_tolerance = -1e-6)
    @test_throws ArgumentError SnoptOptimizer(major_feasibility_tolerance = 0.0)
    @test_throws ArgumentError SnoptOptimizer(minor_feasibility_tolerance = -1.0)
    @test_throws ArgumentError SnoptOptimizer(derivative_option = 0)
    @test_throws ArgumentError SnoptOptimizer(derivative_option = 4)
    @test_throws ArgumentError SnoptOptimizer(hessian = "invalid")
    @test_throws ArgumentError SnoptOptimizer(hessian = "full memory")
    @test_throws ArgumentError SnoptOptimizer(additional_options = ["Major step limit" => 2.0])
    @test_throws ArgumentError SnoptOptimizer(additional_options = Dict("" => 2.0))
    @test_throws ArgumentError SnoptOptimizer(additional_options = Dict("Major step limit" => true))
    @test_throws ArgumentError SnoptOptimizer(additional_options = Dict("Major step limit" => [2.0]))

    if Snopt.has_snopt()
        @test_throws ArgumentError SnoptOptimizer(
            additional_options = Dict("Definitely unknown option" => 1)
        )
        @test_throws ArgumentError SnoptOptimizer(
            additional_options = Dict("Hessian" => "definitely invalid")
        )
    end

    # Valid combinations should construct without error.
    @test SnoptOptimizer() isa SnoptOptimizer
    @test SnoptOptimizer(major_print_level = 0) isa SnoptOptimizer
    @test SnoptOptimizer(derivative_option = 3, hessian = "limited_memory") isa SnoptOptimizer
    @test SnoptOptimizer(additional_options = Dict(
        "Major step limit" => 2.0,
        :linesearch_tolerance => 0.9,
        "Hessian" => "full memory",
    )) isa SnoptOptimizer

    opt = SnoptOptimizer()
    @test !SciMLBase.requireshessian(opt)
    @test !SciMLBase.requiresconshess(opt)
    @test OptimizationSnopt.check_and_convert_maxiters(nothing) === nothing
    @test OptimizationSnopt.check_and_convert_maxiters(7) == 7
    @test_throws ArgumentError OptimizationSnopt.check_and_convert_maxiters(0)
    @test_throws ArgumentError OptimizationSnopt.check_and_convert_maxiters(1.5)
    @test OptimizationSnopt.check_and_convert_maxtime(nothing) === nothing
    @test OptimizationSnopt.check_and_convert_maxtime(2) == 2.0
    @test_throws ArgumentError OptimizationSnopt.check_and_convert_maxtime(Inf)
    @test OptimizationSnopt.map_retcode(33) == SciMLBase.ReturnCode.MaxIters
    @test OptimizationSnopt.snopt_bound_value(Inf) == OptimizationSnopt.SNOPT_BOUND_INF
    @test OptimizationSnopt.snopt_bound_value(-Inf) == -OptimizationSnopt.SNOPT_BOUND_INF
end

@testset "Adapter logging and workspace sizing" begin
    @test OptimizationSnopt.snopt_show_trace(true)
    @test OptimizationSnopt.snopt_show_trace(Val(true))
    @test !OptimizationSnopt.snopt_show_trace(false)
    @test !OptimizationSnopt.snopt_show_trace(Val(false))
    @test !OptimizationSnopt.snopt_show_trace(OptimizationBase.DEFAULT_VERBOSE)

    leniw, lenrw = OptimizationSnopt.snopt_workspace_lengths(100, 100, 10_000)
    @test leniw >= 500 + 100 * (100 + 100)
    @test lenrw >= 500 + 200 * (100 + 100)
    @test leniw > 30_500
    @test lenrw > 40_500

    io = IOBuffer()
    logger = OptimizationSnopt.SnoptProgressLogger(
        false, nothing, true, 2, nothing, Ref(0), io
    )
    @test logger(3, [1.0, 2.0], 4.0, OptimizationBase.MinSense)
    trace = String(take!(io))
    @test occursin("----", trace)
    @test occursin("Eval", trace)
    @test occursin("Major", trace)
    @test occursin("Minor", trace)
    @test occursin("Objective", trace)
    @test occursin("Constr viol", trace)
    @test occursin("Optimality", trace)
    @test occursin("Step", trace)
    @test occursin(r"^\s*1\s+3\s+0\s+4\.00000000e\+00"m, trace)
    @test occursin("4.00000000e+00", trace)
    @test occursin("0.00000000e+00", trace)

    @test logger(4, [1.0, 2.0], 5.0, OptimizationBase.MinSense)
    trace = String(take!(io))
    @test !occursin("Eval", trace)
    @test occursin(r"^\s*2\s+4\s+0\s+5\.00000000e\+00"m, trace)
    @test occursin("5.00000000e+00", trace)

    io = IOBuffer()
    unknown_logger = OptimizationSnopt.SnoptProgressLogger(
        false, nothing, true, 2, nothing, Ref(0), io
    )
    @test unknown_logger((kind = :objective, major_iter = 0, minor_iter = 0,
        x = [1.0, 2.0], f = 4.0))
    trace = String(take!(io))
    @test occursin(r"^\s*1\s+-\s+-\s+4\.00000000e\+00"m, trace)

    raw_color_io = IOBuffer()
    color_io = IOContext(raw_color_io, :color => true)
    OptimizationSnopt.print_trace_algorithm(color_io, SnoptOptimizer())
    color_trace = String(take!(raw_color_io))
    @test occursin("\e[32m", color_trace)
    @test occursin("Algorithm: SnoptOptimizer", color_trace)

    io = IOBuffer()
    ws_rw = zeros(430)
    ws_rw[430] = 0.25
    all_logger = OptimizationSnopt.SnoptProgressLogger(
        false, nothing, true, 2, nothing, Ref(0), io;
        trace_level = OptimizationSnopt.SnoptTraceAll(),
        lb = [0.0, 0.0],
        ub = [2.0, 2.0],
        lcon = [1.0],
        ucon = [1.0],
        store_trace = Val(true),
        ws_rw
    )
    @test all_logger((kind = :objective, major_iter = 1, minor_iter = 2,
        x = [3.0, 1.0], f = 7.0))
    @test all_logger((kind = :constraint, major_iter = 1, minor_iter = 2,
        x = [3.0, 1.0], c = [1.5]))
    OptimizationSnopt.finish_trace!(all_logger, 1, 7.0)
    trace = String(take!(io))
    @test occursin("Minor", trace)
    @test occursin("Constr viol", trace)
    @test occursin("Optimality", trace)
    @test occursin("Bound viol", trace)
    @test occursin("5.00000000e-01", trace)
    @test occursin("2.50000000e-01", trace)
    @test occursin("1.00000000e+00", trace)
    @test occursin("Final", trace)
    stored = OptimizationSnopt.stored_trace(all_logger)
    @test stored isa OptimizationSnopt.SnoptTrace
    @test length(stored.history) == 2
    @test stored.history[1].c == [1.5]
    @test stored.history[1].optimality == 0.25

    if isdefined(Snopt, :SnoptMajorLog)
        io = IOBuffer()
        snlog_logger = OptimizationSnopt.SnoptProgressLogger(
            false, nothing, true, 2, nothing, Ref(0), io;
            trace_from_snlog = true,
            trace_level = OptimizationSnopt.SnoptTraceAll(),
            store_trace = Val(true),
            lb = [0.0, 0.0],
            ub = [2.0, 2.0],
            lcon = [1.0],
            ucon = [1.0]
        )
        event = Snopt.SnoptMajorLog(
            7, 2, 5, 1, 0,
            6.0, 6.5, 1.25, 0.125,
            0.5, 0.25, 0.5, 0.1,
            10.0, 1.0, 0.0, 6.0, 6.5,
            1, 2, 3, 1, 2,
            (false, true),
            [1.5, 0.5, 0.0],
            [1.5],
            [1.5],
            [0.1],
            Int32[0, 0, 0]
        )
        @test snlog_logger(event)
        trace = String(take!(io))
        @test occursin(r"^\s*1\s+2\s+5\s+6\.00000000e\+00"m, trace)
        @test occursin("5.00000000e-01", trace)
        @test occursin("2.50000000e-01", trace)
        @test occursin("1.25000000e-01", trace)
        stored = OptimizationSnopt.stored_trace(snlog_logger)
        @test stored.history[1].x == [1.5, 0.5]
        @test stored.history[1].c == [1.5]
        @test stored.history[1].constraint_violation == 0.5
        @test stored.history[1].optimality == 0.25
    end
end

if !Snopt.has_snopt()
    @info "SNOPT shared library not found — skipping all tests"
    exit(0)
end

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
params = [1.0, 100.0]
l1 = rosenbrock(x0, params)

optfunc = OptimizationFunction((x, p) -> -rosenbrock(x, p), OptimizationBase.AutoZygote())
prob = OptimizationProblem(optfunc, x0, params; sense = OptimizationBase.MaxSense)

callback = function (_, l)
    display(l)
    return false
end

sol = solve(prob, SnoptOptimizer(hessian = "full_memory", major_optimality_tolerance = 1e-8); callback)
@test SciMLBase.successful_retcode(sol)
@test sol ≈ [1, 1]

sol = solve(prob, SnoptOptimizer(hessian = "limited_memory"); callback)
@test SciMLBase.successful_retcode(sol)
@test sol ≈ [1, 1]

function test_hs071(backend, optimizer)
    function objective(x, _)
        return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    end
    function constraints(res, x, _)
        return res .= [
            x[1] * x[2] * x[3] * x[4],
            x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2,
        ]
    end
    prob = OptimizationProblem(
        OptimizationFunction(objective, backend; cons = constraints),
        [1.0, 5.0, 5.0, 1.0];
        sense = OptimizationBase.MinSense,
        lb = [1.0, 1.0, 1.0, 1.0],
        ub = [5.0, 5.0, 5.0, 5.0],
        lcons = [25.0, 40.0],
        ucons = [Inf, 40.0]
    )
    sol = solve(prob, optimizer)
    @test isapprox(sol.objective, 17.014017145179164; atol = 1.0e-5)
    x = [1.0, 4.742999641809297, 3.8211499817883077, 1.3794082897556983]
    @test isapprox(sol.u, x; atol = 1.0e-5)
    @test prod(sol.u) >= 25.0 - 1.0e-5
    @test isapprox(sum(sol.u .^ 2), 40.0; atol = 1.0e-5)
    return
end

@testset "backends" begin
    backends = (
        AutoForwardDiff(),
        AutoFiniteDiff(),
        AutoSparse(AutoForwardDiff()),
    )
    for backend in backends
        @testset "$backend" begin
            test_hs071(backend, SnoptOptimizer())
        end
    end
end

include("additional_tests.jl")
include("advanced_features.jl")
include("problem_types.jl")

@testset "HS71 with tight tolerances" begin
    function hs71_obj(x, _)
        x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    end
    function hs71_cons(res, x, _)
        res[1] = x[1] * x[2] * x[3] * x[4]
        res[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    end
    prob = OptimizationProblem(
        OptimizationFunction(hs71_obj, AutoForwardDiff(); cons = hs71_cons),
        [1.0, 5.0, 5.0, 1.0], nothing;
        lb = [1.0, 1.0, 1.0, 1.0],
        ub = [5.0, 5.0, 5.0, 5.0],
        lcons = [25.0, 40.0],
        ucons = [Inf, 40.0]
    )
    sol = solve(prob, SnoptOptimizer(
        major_optimality_tolerance = 1.0e-9,
        major_feasibility_tolerance = 1.0e-9
    ))
    @test SciMLBase.successful_retcode(sol)
    @test sol.objective ≈ 17.014017 atol = 1.0e-5
end

@testset "Cache interface (init/solve!)" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    prob = OptimizationProblem(
        OptimizationFunction(rosenbrock, AutoForwardDiff()),
        zeros(2), [1.0, 100.0]
    )
    cache = init(prob, SnoptOptimizer())
    @test cache isa OptimizationSnopt.SnoptCache
    opt_setup, summfile, reader_task, show_output, logger = OptimizationSnopt.map_optimizer_args(cache, cache.opt)
    @test opt_setup isa Snopt.SnoptB
    @test all(isfinite, opt_setup.bl)
    @test all(isfinite, opt_setup.bu)
    @test opt_setup.bl[end] == -OptimizationSnopt.SNOPT_BOUND_INF
    @test opt_setup.bu[end] == OptimizationSnopt.SNOPT_BOUND_INF
    @test isnothing(reader_task)
    @test summfile == ""
    @test !show_output
    @test logger isa OptimizationSnopt.SnoptProgressLogger
    finalize(opt_setup.ws)
    sol = solve!(cache)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
end

@testset "Common interface arguments" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    prob = OptimizationProblem(
        OptimizationFunction(rosenbrock, AutoForwardDiff()),
        zeros(2), [1.0, 100.0]
    )

    sol1 = solve(prob, SnoptOptimizer(); abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol1)
    @test sol1.u ≈ [1.0, 1.0] atol = 1.0e-8

    sol2 = solve(prob, SnoptOptimizer(); maxiters = 5)
    @test sol2.stats.iterations <= 5

    sol_time = solve(prob, SnoptOptimizer(); maxtime = 10.0)
    @test SciMLBase.successful_retcode(sol_time)

    sol3 = solve(prob, SnoptOptimizer(); verbose = false)
    @test sol3 isa SciMLBase.OptimizationSolution

    sol_trace = solve(prob, SnoptOptimizer(); store_trace = Val(true),
        trace_level = OptimizationSnopt.SnoptTraceMinimal(2))
    @test SciMLBase.successful_retcode(sol_trace)
    @test sol_trace.trace isa OptimizationSnopt.SnoptTrace
    @test sol_trace.original.trace === sol_trace.trace
    @test sol_trace.trace.history[end].iteration == -1

    sol4 = solve(prob, SnoptOptimizer(); verbose = Val(true))
    @test sol4 isa SciMLBase.OptimizationSolution
end

@testset "additional_options passthrough" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    prob = OptimizationProblem(
        OptimizationFunction(rosenbrock, AutoForwardDiff()),
        zeros(2), [1.0, 100.0]
    )

    opt = SnoptOptimizer(
        additional_options = Dict{String, Any}(
            "Major step limit" => 2.0,
            "Linesearch tolerance" => 0.9
        )
    )
    sol = solve(prob, opt)
    @test SciMLBase.successful_retcode(sol)
end
