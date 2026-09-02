# Hock-Schittkowski problem 71 is a constrained nonlinear benchmark.
#
# Minimize:
#   f(x) = x₁x₄(x₁ + x₂ + x₃) + x₃
#
# Subject to:
#   x₁x₂x₃x₄ ≥ 25
#   x₁² + x₂² + x₃² + x₄² = 40
#   1 ≤ xᵢ ≤ 5
#
# Run from the OptimizationSNOPT.jl repository root:
#   julia --project=examples examples/hs71.jl

using ADTypes: AutoFiniteDiff, AutoForwardDiff
using OptimizationSNOPT

function hs71_obj(x, _)
    x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
end

function hs71_cons(res, x, _)
    res[1] = x[1] * x[2] * x[3] * x[4]
    res[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
end

x0 = [1.0, 5.0, 5.0, 1.0]

# AutoForwardDiff computes exact derivatives through Julia code.
prob_fd = OptimizationProblem(
    OptimizationFunction(hs71_obj, AutoForwardDiff(); cons = hs71_cons),
    x0, nothing;
    lb = [1.0, 1.0, 1.0, 1.0],
    ub = [5.0, 5.0, 5.0, 5.0],
    lcons = [25.0, 40.0],
    ucons = [Inf, 40.0],
)

alg = SnoptOptimizer(
    major_iterations_limit = 500,
    major_optimality_tolerance = 1e-8,
    major_feasibility_tolerance = 1e-8,
)

sol_fd = solve(prob_fd, alg; show_trace = Val(true))

println("ForwardDiff")
println("  x        = ", round.(sol_fd.u; digits = 4))
println("  objective = ", round(sol_fd.objective; digits = 6))
println("  return code = ", sol_fd.retcode)
println("  SNOPT status = ", sol_fd.original.inform)

# AutoFiniteDiff estimates derivatives from nearby function values.
prob_fin = OptimizationProblem(
    OptimizationFunction(hs71_obj, AutoFiniteDiff(); cons = hs71_cons),
    x0, nothing;
    lb = [1.0, 1.0, 1.0, 1.0],
    ub = [5.0, 5.0, 5.0, 5.0],
    lcons = [25.0, 40.0],
    ucons = [Inf, 40.0],
)
sol_fin = solve(prob_fin, alg; show_trace = Val(true))

println()
println("FiniteDiff")
println("  x = ", round.(sol_fin.u; digits = 4))
println("  objective = ", round(sol_fin.objective; digits = 6))
println("  return code = ", sol_fin.retcode)

# These functions provide derivatives directly.
function hs71_grad!(g, x, _)
    g[1] = x[4] * (2x[1] + x[2] + x[3])
    g[2] = x[1] * x[4]
    g[3] = x[1] * x[4] + 1.0
    g[4] = x[1] * (x[1] + x[2] + x[3])
end

function hs71_cons_j!(J, x, _)
    J[1, 1] = x[2] * x[3] * x[4]
    J[1, 2] = x[1] * x[3] * x[4]
    J[1, 3] = x[1] * x[2] * x[4]
    J[1, 4] = x[1] * x[2] * x[3]
    J[2, 1] = 2x[1]
    J[2, 2] = 2x[2]
    J[2, 3] = 2x[3]
    J[2, 4] = 2x[4]
end

prob_ana = OptimizationProblem(
    OptimizationFunction(hs71_obj, AutoForwardDiff();
        grad = hs71_grad!,
        cons = hs71_cons,
        cons_j = hs71_cons_j!),
    x0, nothing;
    lb = [1.0, 1.0, 1.0, 1.0],
    ub = [5.0, 5.0, 5.0, 5.0],
    lcons = [25.0, 40.0],
    ucons = [Inf, 40.0],
)

sol_ana = solve(prob_ana, alg; show_trace = Val(true))

println()
println("Provided derivatives")
println("  x = ", round.(sol_ana.u; digits = 4))
println("  objective = ", round(sol_ana.objective; digits = 6))
println("  return code = ", sol_ana.retcode)

println()
println("Expected x ≈ [1.0000, 4.7430, 3.8211, 1.3794]")
println("Expected objective ≈ 17.014017")
