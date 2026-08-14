## examples/hs71.jl
##
## Hock-Schittkowski problem 71 — a standard NLP benchmark with both
## inequality and equality constraints.
##
## Problem:
##   minimize    f(x) = x₁ x₄ (x₁ + x₂ + x₃) + x₃
##   subject to  g₁(x) = x₁ x₂ x₃ x₄ ≥ 25       (inequality)
##               g₂(x) = x₁² + x₂² + x₃² + x₄² = 40  (equality)
##               1 ≤ xᵢ ≤ 5,  i = 1,…,4
##
## Known solution:
##   x* ≈ [1.000, 4.743, 3.821, 1.379]
##   f* ≈ 17.0140
##
## Run:
##   julia --project=examples/ examples/hs71.jl
using Pkg
Pkg.activate(@__DIR__)

using OptimizationBase
using OptimizationSNOPT
using ADTypes, ForwardDiff, FiniteDiff

# Objective
function hs71_obj(x, _)
    x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
end

# Constraints (returned as a vector; bounds defined separately below)
#   g₁(x) = x₁ x₂ x₃ x₄      ∈ [25, ∞)
#   g₂(x) = x₁² + x₂² + x₃² + x₄²  ∈ [40, 40]
function hs71_cons(res, x, _)
    res[1] = x[1] * x[2] * x[3] * x[4]
    res[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
end

# Initial point
x0 = [1.0, 5.0, 5.0, 1.0]

# Solve with ForwardDiff
prob_fd = OptimizationProblem(
    OptimizationFunction(hs71_obj, AutoForwardDiff(); cons = hs71_cons),
    x0, nothing;
    lb     = [1.0, 1.0, 1.0, 1.0],
    ub     = [5.0, 5.0, 5.0, 5.0],
    lcons  = [25.0, 40.0],   # lower bounds on constraints
    ucons  = [Inf,  40.0],   # upper bounds (Inf = one-sided inequality)
)

alg = SnoptOptimizer(
    major_iterations_limit = 500,
    major_optimality_tolerance = 1e-8,
    major_feasibility_tolerance = 1e-8,
)

sol_fd = solve(prob_fd, alg; show_trace=Val(true))

println("─── ForwardDiff backend ────────────────────")
println("  x*       = ", round.(sol_fd.u; digits=4))
println("  f(x*)    = ", round(sol_fd.objective; digits=6))
println("  retcode  = ", sol_fd.retcode)
println("  inform   = ", sol_fd.original.inform)

# ── 4. Solve with finite differences (FiniteDiff) ─────────────────────────────
prob_fin = OptimizationProblem(
    OptimizationFunction(hs71_obj, AutoFiniteDiff(); cons = hs71_cons),
    x0, nothing;
    lb     = [1.0, 1.0, 1.0, 1.0],
    ub     = [5.0, 5.0, 5.0, 5.0],
    lcons  = [25.0, 40.0],
    ucons  = [Inf,  40.0],
)
sol_fin = solve(prob_fin, alg; show_trace=Val(true))

println()
println("─── FiniteDiff backend ─────────────────────")
println("  x*       = ", round.(sol_fin.u; digits=4))
println("  f(x*)    = ", round(sol_fin.objective; digits=6))
println("  retcode  = ", sol_fin.retcode)

# ── 5. Solve with analytical gradients ────────────────────────────────────────

function hs71_grad!(g, x, _)
    g[1] = x[4] * (2x[1] + x[2] + x[3])
    g[2] = x[1] * x[4]
    g[3] = x[1] * x[4] + 1.0
    g[4] = x[1] * (x[1] + x[2] + x[3])
end

# Constraint Jacobian: row i = gradient of constraint i
function hs71_cons_j!(J, x, _)
    # ∂g₁/∂x
    J[1, 1] = x[2] * x[3] * x[4]
    J[1, 2] = x[1] * x[3] * x[4]
    J[1, 3] = x[1] * x[2] * x[4]
    J[1, 4] = x[1] * x[2] * x[3]
    # ∂g₂/∂x
    J[2, 1] = 2x[1]
    J[2, 2] = 2x[2]
    J[2, 3] = 2x[3]
    J[2, 4] = 2x[4]
end

prob_ana = OptimizationProblem(
    OptimizationFunction(hs71_obj, AutoForwardDiff();
                         grad   = hs71_grad!,
                         cons   = hs71_cons,
                         cons_j = hs71_cons_j!),
    x0, nothing;
    lb     = [1.0, 1.0, 1.0, 1.0],
    ub     = [5.0, 5.0, 5.0, 5.0],
    lcons  = [25.0, 40.0],
    ucons  = [Inf,  40.0],
)

sol_ana = solve(prob_ana, alg; show_trace=Val(true))

println()
println("─── Analytical gradients ───────────────────")
println("  x*       = ", round.(sol_ana.u; digits=4))
println("  f(x*)    = ", round(sol_ana.objective; digits=6))
println("  retcode  = ", sol_ana.retcode)

println()
println("Expected:  x* ≈ [1.0000, 4.7430, 3.8211, 1.3794]")
println("           f* ≈ 17.014017")
