using OptimizationBase, OptimizationSNOPT
using Test
using LinearAlgebra
using ADTypes, ForwardDiff

@testset "Additional SNOPT Examples" begin
    @testset "Simple 2D Example" begin
        # minimize -x[1] - x[2]
        # s.t. x[2] - x[1]^2 = 0,  -1 <= x[1] <= 1

        function simple_objective(x, p)
            return -x[1] - x[2]
        end

        function simple_constraint(res, x, p)
            res[1] = x[2] - x[1]^2
        end

        optfunc = OptimizationFunction(
            simple_objective, AutoForwardDiff();
            cons = simple_constraint
        )
        prob = OptimizationProblem(
            optfunc, [0.0, 0.0], nothing;
            lb = [-1.0, -Inf],
            ub = [1.0, Inf],
            lcons = [0.0],
            ucons = [0.0]
        )
        sol = solve(prob, SnoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test sol.u[1] ≈ 1.0 atol = 1.0e-5
        @test sol.u[2] ≈ 1.0 atol = 1.0e-5
        @test sol.objective ≈ -2.0 atol = 1.0e-5
    end

    @testset "Luksan-Vlcek Problem 1" begin
        function lv1_objective(x, p)
            n = length(x)
            obj = 0.0
            for i in 1:(n - 1)
                obj += 100 * (x[i]^2 - x[i + 1])^2 + (x[i] - 1)^2
            end
            return obj
        end

        function lv1_constraints(res, x, p)
            n = length(x)
            for i in 1:(n - 2)
                res[i] = 3 * x[i + 1]^3 + 2 * x[i + 2] - 5 +
                    sin(x[i + 1] - x[i + 2]) * sin(x[i + 1] + x[i + 2]) +
                    4 * x[i + 1] - x[i] * exp(x[i] - x[i + 1]) - 3
            end
        end

        n = 5
        x0 = fill(3.0, n)

        optfunc = OptimizationFunction(
            lv1_objective, AutoForwardDiff();
            cons = lv1_constraints
        )
        prob = OptimizationProblem(
            optfunc, x0, nothing;
            lcons = fill(4.0, n - 2),
            ucons = fill(6.0, n - 2)
        )
        sol = solve(prob, SnoptOptimizer(); maxiters = 1000)

        @test SciMLBase.successful_retcode(sol)
        @test length(sol.u) == n
        res = zeros(n - 2)
        lv1_constraints(res, sol.u, nothing)
        @test all(res .- 6 .<= 0 .|| res .≈ 6)
        @test all(res .- 4 .>= 0 .|| res .≈ 4)
    end

    @testset "Bound Constrained Quadratic" begin
        # minimize (x-2)^2 + (y-3)^2, s.t. 0 <= x <= 1, 0 <= y <= 2
        quadratic(x, p) = (x[1] - 2)^2 + (x[2] - 3)^2

        optfunc = OptimizationFunction(quadratic, AutoForwardDiff())
        prob = OptimizationProblem(
            optfunc, [0.5, 1.0], nothing;
            lb = [0.0, 0.0],
            ub = [1.0, 2.0]
        )
        sol = solve(prob, SnoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test sol.u[1] ≈ 1.0 atol = 1.0e-5
        @test sol.u[2] ≈ 2.0 atol = 1.0e-5
    end

    @testset "Nonlinear Least Squares" begin
        t = [0.0, 0.5, 1.0, 1.5, 2.0]
        y_data = [1.0, 1.5, 2.1, 3.2, 4.8]

        function nls_objective(x, p)
            sum_sq = 0.0
            for i in 1:length(t)
                pred = x[1] * exp(x[2] * t[i])
                sum_sq += (y_data[i] - pred)^2
            end
            return sum_sq
        end

        optfunc = OptimizationFunction(nls_objective, AutoForwardDiff())
        prob = OptimizationProblem(optfunc, [1.0, 0.5], nothing)
        sol = solve(prob, SnoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test sol.objective < 0.1
    end

    @testset "Mixed Integer-like Problem (Relaxed)" begin
        # SNOPT is continuous; integer metadata follows OptimizationIpopt's
        # convention here and solves the relaxed problem.
        # minimize x^2 + y^2, s.t. x + y >= 3.5, 0 <= x,y <= 5
        objective(x, p) = x[1]^2 + x[2]^2

        function constraint(res, x, p)
            res[1] = x[1] + x[2]
        end

        optfunc = OptimizationFunction(
            objective, AutoForwardDiff();
            cons = constraint
        )
        prob = OptimizationProblem(
            optfunc, [2.0, 2.0], nothing;
            lb = [0.0, 0.0],
            ub = [5.0, 5.0],
            int = [true, true],
            lcons = [3.5],
            ucons = [Inf]
        )
        sol = solve(prob, SnoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test sol.u[1] + sol.u[2] ≈ 3.5 atol = 1.0e-5
        @test sol.u[1] ≈ sol.u[2] atol = 1.0e-5
    end

    @testset "Barrier Method Test" begin
        # minimize -log(x) - log(y), s.t. x + y <= 2, x,y > 0
        function barrier_objective(x, p)
            return -log(x[1]) - log(x[2])
        end

        function barrier_constraint(res, x, p)
            res[1] = x[1] + x[2]
        end

        optfunc = OptimizationFunction(
            barrier_objective, AutoForwardDiff();
            cons = barrier_constraint
        )
        prob = OptimizationProblem(
            optfunc, [0.5, 0.5], nothing;
            lb = [1.0e-6, 1.0e-6],
            ub = [Inf, Inf],
            lcons = [-Inf],
            ucons = [2.0]
        )
        sol = solve(prob, SnoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test sol.u[1] + sol.u[2] ≈ 2.0 atol = 1.0e-4
        @test sol.u[1] ≈ 1.0 atol = 1.0e-4
        @test sol.u[2] ≈ 1.0 atol = 1.0e-4
    end

    @testset "Large Scale Sparse Problem" begin
        # minimize sum(x[i]^2) + sum((x[i]-x[i+1])^2), s.t. x[1]+x[n] >= 1
        n = 20

        function sparse_objective(x, p)
            obj = sum(x[i]^2 for i in 1:n)
            obj += sum((x[i] - x[i + 1])^2 for i in 1:(n - 1))
            return obj
        end

        function sparse_constraint(res, x, p)
            res[1] = x[1] + x[n]
        end

        optfunc = OptimizationFunction(
            sparse_objective, AutoForwardDiff();
            cons = sparse_constraint
        )
        x0 = fill(0.1, n)
        prob = OptimizationProblem(
            optfunc, x0, nothing;
            lcons = [1.0],
            ucons = [Inf]
        )
        sol = solve(prob, SnoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test sol.u[1] + sol.u[n] >= 1.0 - 1.0e-5
    end
end

@testset "Hessian Approximation Options" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

    x0 = [0.0, 0.0]
    p = [1.0, 100.0]

    @testset "Hessian full memory (default)" begin
        optfunc = OptimizationFunction(rosenbrock, AutoForwardDiff())
        prob = OptimizationProblem(optfunc, x0, p)
        sol = solve(prob, SnoptOptimizer(hessian = "full_memory", major_optimality_tolerance = 1e-8))

        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [1.0, 1.0]
    end

    @testset "Hessian limited memory" begin
        optfunc = OptimizationFunction(rosenbrock, AutoForwardDiff())
        prob = OptimizationProblem(optfunc, x0, p)
        sol = solve(prob, SnoptOptimizer(hessian = "limited_memory", major_optimality_tolerance = 1e-8))

        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [1.0, 1.0]
    end
end

@testset "Warm Start Tests" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

    x0 = [0.5, 0.5]
    p = [1.0, 100.0]

    optfunc = OptimizationFunction(rosenbrock, AutoForwardDiff())
    prob = OptimizationProblem(optfunc, x0, p)

    cache = init(prob, SnoptOptimizer())
    sol1 = solve!(cache)

    @test SciMLBase.successful_retcode(sol1)
    @test sol1.u ≈ [1.0, 1.0] atol = 1.0e-4
end
