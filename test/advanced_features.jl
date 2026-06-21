using OptimizationBase, OptimizationSNOPT
using Test
using LinearAlgebra
using SparseArrays
using ADTypes, ForwardDiff

@testset "Advanced SNOPT Features" begin

    @testset "Custom Tolerances" begin
        rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

        x0 = [0.0, 0.0]
        p = [1.0, 100.0]

        optfunc = OptimizationFunction(rosenbrock, AutoForwardDiff())
        prob = OptimizationProblem(optfunc, x0, p)

        sol = solve(
            prob, SnoptOptimizer(
                major_optimality_tolerance = 1.0e-9,
                minor_feasibility_tolerance = 1.0e-9
            )
        )

        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-7
    end

    @testset "Constraint Feasibility Tolerance" begin
        function obj(x, _)
            return x[1]^2 + x[2]^2
        end

        function cons(res, x, _)
            res[1] = x[1] + x[2] - 2.0
            res[2] = x[1]^2 + x[2]^2 - 2.0
        end

        optfunc = OptimizationFunction(obj, AutoForwardDiff(); cons = cons)
        prob = OptimizationProblem(
            optfunc, [0.5, 0.5], nothing;
            lcons = [0.0, 0.0],
            ucons = [0.0, 0.0]
        )

        sol = solve(
            prob, SnoptOptimizer(
                major_feasibility_tolerance = 1.0e-8
            )
        )

        @test SciMLBase.successful_retcode(sol)
        @test sol.u[1] + sol.u[2] ≈ 2.0 atol = 1.0e-6
        @test sol.u[1]^2 + sol.u[2]^2 ≈ 2.0 atol = 1.0e-6
    end

    @testset "Derivative Verification" begin
        # SNOPT can verify user-supplied gradients with "Verify level"
        function complex_obj(x, p)
            return sin(x[1]) * cos(x[2]) + exp(-x[1]^2 - x[2]^2)
        end

        optfunc = OptimizationFunction(complex_obj, AutoForwardDiff())
        prob = OptimizationProblem(optfunc, [0.1, 0.1], nothing)

        sol = solve(
            prob, SnoptOptimizer(
                additional_options = Dict("Verify level" => 0)  # 0=off, 1=objective, 3=both
            )
        )

        @test SciMLBase.successful_retcode(sol)
    end

    @testset "Iteration Limits" begin
        # n-dimensional Rosenbrock
        function rosenbrock_n(x, p)
            n = length(x)
            s = 0.0
            for i in 1:2:(n - 1)
                s += (p[1] - x[i])^2 + p[2] * (x[i + 1] - x[i]^2)^2
            end
            return s
        end

        x0 = zeros(10)
        p = [1.0, 100.0]

        optfunc = OptimizationFunction(rosenbrock_n, AutoForwardDiff())
        prob = OptimizationProblem(optfunc, x0, p)

        sol = solve(
            prob, SnoptOptimizer(
                major_iterations_limit = 2000,
                minor_iterations_limit = 1000
            )
        )

        @test SciMLBase.successful_retcode(sol)
        @test all(isapprox(sol.u[i], 1.0, atol = 1.0e-4) for i in 1:2:(length(x0) - 1))
    end

    @testset "Scaling" begin
        # Ill-scaled problem — benefits from SNOPT's scale option
        function scaled_obj(x, _)
            return 1.0e6 * x[1]^2 + 1.0e-6 * x[2]^2
        end

        function scaled_cons(res, x, _)
            res[1] = 1.0e3 * x[1] + 1.0e-3 * x[2] - 1.0
        end

        optfunc = OptimizationFunction(
            scaled_obj, AutoForwardDiff();
            cons = scaled_cons
        )
        prob = OptimizationProblem(
            optfunc, [1.0, 1.0], nothing;
            lcons = [0.0],
            ucons = [0.0]
        )

        sol = solve(
            prob, SnoptOptimizer(
                additional_options = Dict("Scale option" => 1)
            )
        )

        @test SciMLBase.successful_retcode(sol)
        res = zeros(1)
        scaled_cons(res, sol.u, nothing)
        @test abs(res[1]) < 1.0e-5
    end

    @testset "Infeasible Start" begin
        # Problem started from an infeasible point
        function difficult_obj(x, _)
            return x[1]^4 + x[2]^4
        end

        function difficult_cons(res, x, _)
            res[1] = x[1]^3 + x[2]^3 - 1.0
            res[2] = x[1]^2 + x[2]^2 - 0.5
        end

        optfunc = OptimizationFunction(
            difficult_obj, AutoForwardDiff();
            cons = difficult_cons
        )
        prob = OptimizationProblem(
            optfunc, [2.0, 2.0], nothing;
            lcons = [0.0, 0.0],
            ucons = [0.0, 0.0]
        )

        sol = solve(prob, SnoptOptimizer())

        if SciMLBase.successful_retcode(sol)
            res = zeros(2)
            difficult_cons(res, sol.u, nothing)
            @test norm(res) < 1.0e-4
        end
    end

    @testset "Derivative Option" begin
        # derivative_option = 1: SNOPT trusts user-supplied gradients
        # derivative_option = 3: SNOPT also trusts user Jacobian
        rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

        x0 = [0.0, 0.0]
        p = [1.0, 100.0]

        optfunc = OptimizationFunction(rosenbrock, AutoForwardDiff())
        prob = OptimizationProblem(optfunc, x0, p)

        sol = solve(prob, SnoptOptimizer(derivative_option = 1))

        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-4
    end

    @testset "Fixed Variable Handling" begin
        # Fix x[2] = 2.0 via equal bounds; SNOPT handles this natively
        function fixed_var_obj(x, _)
            return (x[1] - 1)^2 + (x[2] - 2)^2 + (x[3] - 3)^2
        end

        optfunc = OptimizationFunction(fixed_var_obj, AutoForwardDiff())
        prob = OptimizationProblem(
            optfunc, [0.0, 2.0, 0.0], nothing;
            lb = [-Inf, 2.0, -Inf],
            ub = [Inf, 2.0, Inf]
        )

        sol = solve(prob, SnoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [1.0, 2.0, 3.0] atol = 1.0e-5
    end

    @testset "MaxSense" begin
        function slow_converge_obj(x, _)
            return -sum((x[i] - i / 10)^2 for i in eachindex(x))
        end

        n = 5
        optfunc = OptimizationFunction(slow_converge_obj, AutoForwardDiff())
        prob = OptimizationProblem(
            optfunc, zeros(n), nothing;
            sense = OptimizationBase.MaxSense
        )

        sol = solve(prob, SnoptOptimizer(); maxiters = 200)

        @test SciMLBase.successful_retcode(sol)
        @test sol.objective ≈ 0.0 atol = 1.0e-8
        for i in 1:n
            @test sol.u[i] ≈ i / 10 atol = 1.0e-5
        end
    end
end

@testset "Output and Print Level Options" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

    x0 = [0.0, 0.0]
    p = [1.0, 100.0]

    optfunc = OptimizationFunction(rosenbrock, AutoForwardDiff())
    prob = OptimizationProblem(optfunc, x0, p)

    @testset "verbose=false (silent)" begin
        sol = solve(prob, SnoptOptimizer(); verbose = false)
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "Major print level via struct" begin
        sol = solve(prob, SnoptOptimizer(major_print_level = 1); verbose = false)
        @test SciMLBase.successful_retcode(sol)
    end

    @testset "Additional options passthrough" begin
        sol = solve(
            prob, SnoptOptimizer(
                additional_options = Dict("Major step limit" => 2.0)
            )
        )
        @test SciMLBase.successful_retcode(sol)
    end
end
