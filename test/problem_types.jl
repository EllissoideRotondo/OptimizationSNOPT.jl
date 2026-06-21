using OptimizationBase, OptimizationSNOPT
using Test
using LinearAlgebra
using SparseArrays
using ADTypes, ForwardDiff

@testset "Specific Problem Types" begin

    @testset "Optimal Control Problem" begin
        # Discretized: minimize control effort subject to linear dynamics
        N = 20
        dt = 0.1

        function control_objective(z, _)
            u_start = N + 1
            return sum(z[i]^2 for i in u_start:length(z))
        end

        function dynamics_constraints(res, z, _)
            for i in 1:(N - 1)
                res[i] = z[i + 1] - z[i] - dt * z[N + i]
            end
            res[N]     = z[1] - 0.0
            res[N + 1] = z[N] - 1.0
        end

        n_vars = N + (N - 1)
        n_cons = N + 1

        optfunc = OptimizationFunction(
            control_objective, AutoForwardDiff();
            cons = dynamics_constraints
        )
        z0 = zeros(n_vars)
        prob = OptimizationProblem(
            optfunc, z0;
            lcons = zeros(n_cons),
            ucons = zeros(n_cons)
        )

        sol = solve(prob, SnoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test sol.u[1] ≈ 0.0 atol = 1.0e-5
        @test sol.u[N] ≈ 1.0 atol = 1.0e-5
    end

    @testset "Portfolio Optimization" begin
        n_assets = 5
        μ = [0.05, 0.1, 0.15, 0.08, 0.12]
        Σ = [
            0.05 0.01 0.02 0.01 0.0;
            0.01 0.1  0.03 0.02 0.01;
            0.02 0.03 0.15 0.02 0.03;
            0.01 0.02 0.02 0.08 0.02;
            0.0  0.01 0.03 0.02 0.06
        ]
        target_return = 0.1

        function portfolio_risk(w, _)
            return dot(w, Σ * w)
        end

        function portfolio_constraints(res, w, _)
            res[1] = sum(w) - 1.0
            res[2] = dot(μ, w) - target_return
        end

        optfunc = OptimizationFunction(
            portfolio_risk, AutoForwardDiff();
            cons = portfolio_constraints
        )
        w0 = fill(1.0 / n_assets, n_assets)
        prob = OptimizationProblem(
            optfunc, w0;
            lb = zeros(n_assets),
            ub = ones(n_assets),
            lcons = [0.0, 0.0],
            ucons = [0.0, Inf]
        )

        sol = solve(prob, SnoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test sum(sol.u) ≈ 1.0 atol = 1.0e-5
        @test dot(μ, sol.u) >= target_return - 1.0e-5
        @test all(sol.u .>= -1.0e-5)
    end

    @testset "Geometric Programming" begin
        function geometric_obj(x, _)
            return exp(x[1]) * exp(x[2]) * exp(x[3])
        end

        function geometric_cons(res, x, _)
            res[1] = exp(2 * x[1] + x[2] - x[3]) - 1.0
            res[2] = exp(x[1]) + exp(x[2]) + exp(x[3]) - 3.0
        end

        optfunc = OptimizationFunction(
            geometric_obj, AutoForwardDiff();
            cons = geometric_cons
        )
        x0 = zeros(3)
        prob = OptimizationProblem(
            optfunc, x0;
            lcons = [-Inf, 0.0],
            ucons = [0.0, 0.0]
        )

        sol = solve(prob, SnoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        res = zeros(2)
        geometric_cons(res, sol.u, nothing)
        @test res[1] <= 1.0e-5
        @test abs(res[2]) <= 1.0e-5
    end

    @testset "Parameter Estimation" begin
        true_params = [2.0, 0.5, 0.1]
        t_data = collect(0:0.5:5)
        y_data = @. true_params[1] * exp(-true_params[2] * t_data) + true_params[3]
        y_data += [0.05, 0.01, 0.01, 0.025, 0.0001, 0.004, 0.0056, 0.003, 0.0076, 0.012, 0.0023]

        function residual_sum_squares(params, _)
            a, b, c = params
            residuals = @. y_data - (a * exp(-b * t_data) + c)
            return sum(residuals .^ 2)
        end

        optfunc = OptimizationFunction(residual_sum_squares, AutoForwardDiff())
        params0 = [1.0, 1.0, 0.0]
        prob = OptimizationProblem(
            optfunc, params0;
            lb = [0.0, 0.0, -1.0],
            ub = [10.0, 10.0, 1.0]
        )

        sol = solve(
            prob, SnoptOptimizer(
                major_optimality_tolerance = 1.0e-10
            )
        )

        @test SciMLBase.successful_retcode(sol)
        @test sol.u[1] ≈ true_params[1] atol = 0.2
        @test sol.u[2] ≈ true_params[2] atol = 0.1
        @test sol.u[3] ≈ true_params[3] atol = 0.05
    end

    @testset "Network Flow Problem" begin
        costs      = [2.0, 3.0, 1.0, 4.0, 2.0]
        capacities = [10.0, 8.0, 5.0, 10.0, 10.0]
        required_flow = 15.0

        function flow_cost(flows, _)
            return dot(costs, flows)
        end

        function flow_constraints(res, flows, _)
            res[1] = flows[1] - flows[3] - flows[4]
            res[2] = flows[2] + flows[3] - flows[5]
            res[3] = flows[1] + flows[2] - required_flow
            res[4] = flows[4] + flows[5] - required_flow
        end

        optfunc = OptimizationFunction(
            flow_cost, AutoForwardDiff();
            cons = flow_constraints
        )
        flows0 = fill(required_flow / 2, 5)
        prob = OptimizationProblem(
            optfunc, flows0, nothing;
            lb = zeros(5),
            ub = capacities,
            lcons = zeros(4),
            ucons = zeros(4)
        )

        sol = solve(prob, SnoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test all(sol.u .>= -1.0e-5)
        @test all(sol.u .<= capacities .+ 1.0e-5)
        res = zeros(4)
        flow_constraints(res, sol.u, nothing)
        @test norm(res) < 1.0e-5
    end

    @testset "Robust Optimization" begin
        function robust_objective(x, _)
            return sum(x .^ 2) + sum(abs.(x))
        end

        function robust_constraints(res, x, _)
            res[1] = sum(x) - 1.0
        end

        n = 3
        optfunc = OptimizationFunction(
            robust_objective, AutoForwardDiff();
            cons = robust_constraints
        )
        x0 = fill(1.0 / n, n)
        prob = OptimizationProblem(
            optfunc, x0, nothing;
            lcons = [0.0],
            ucons = [Inf]
        )

        sol = solve(prob, SnoptOptimizer())

        @test SciMLBase.successful_retcode(sol)
        @test sum(sol.u) >= 1.0 - 1.0e-5
    end
end

@testset "Stress Tests" begin
    @testset "High-dimensional Problem" begin
        n = 100
        A = randn(n, n)
        Q = A' * A + I
        b = randn(n)

        function large_quadratic(x, _)
            return 0.5 * dot(x, Q * x) - dot(b, x)
        end

        optfunc = OptimizationFunction(large_quadratic, AutoForwardDiff())
        x0 = randn(n)
        prob = OptimizationProblem(optfunc, x0)

        sol = solve(prob, SnoptOptimizer(); maxiters = 1000)

        @test SciMLBase.successful_retcode(sol)
        grad = Q * sol.u - b
        @test norm(grad) < 1.0e-4
    end

    @testset "Highly Nonlinear Problem" begin
        function trig_objective(x, _)
            n = length(x)
            return sum(
                sin(x[i])^2 * cos(x[i])^2 + exp(-abs(x[i] - π / 4)) for i in 1:n
            )
        end

        n = 10
        optfunc = OptimizationFunction(trig_objective, AutoForwardDiff())
        x0 = randn(n)
        prob = OptimizationProblem(
            optfunc, x0;
            lb = fill(-2π, n),
            ub = fill(2π, n)
        )

        sol = solve(prob, SnoptOptimizer(hessian = "limited_memory"))

        @test SciMLBase.successful_retcode(sol)
    end
end
