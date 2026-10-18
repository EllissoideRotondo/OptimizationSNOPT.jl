```@meta
CurrentModule = OptimizationSNOPT
```

# Getting started

In this tutorial, we solve two problems with SNOPT through Optimization.jl. The first
has no constraints; the second adds variable bounds and nonlinear constraints. Both
examples need a working SNOPT library (see [Installation](installation.md)).

## Solve an unconstrained problem

We minimize the Rosenbrock function

```math
f(x, p) = (p_1 - x_1)^2 + p_2 (x_2 - x_1^2)^2,
```

with parameters ``p = (1, 100)``. The minimum is at ``x = (1, 1)`` with ``f = 0``.

We define the objective with the signature `f(x, p)` that Optimization.jl uses, and
choose `AutoForwardDiff()` so that ForwardDiff.jl computes the gradient that SNOPT
needs. `OptimizationSNOPT` re-exports `OptimizationProblem`, `OptimizationFunction`,
`solve` and the automatic differentiation selectors, so no other package is loaded:

```jldoctest rosenbrock
using OptimizationSNOPT

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

f = OptimizationFunction(rosenbrock, AutoForwardDiff())
prob = OptimizationProblem(f, [0.0, 0.0], [1.0, 100.0])
sol = solve(prob, SnoptOptimizer())
sol.retcode

# output

ReturnCode.Success = 1
```

## Read the solution

`solve` returns an Optimization.jl solution. The return code above is
`ReturnCode.Success`, so the point `sol.u` is the solution:

```jldoctest rosenbrock
round.(sol.u; digits = 6)

# output

2-element Vector{Float64}:
 1.0
 1.0
```

| Field | Content |
| --- | --- |
| `sol.retcode` | Optimization.jl return code, for example `ReturnCode.Success`. |
| `sol.u` | Solution. |
| `sol.objective` | Objective value at `sol.u`. |
| `sol.original.inform` | SNOPT status code, an integer. |
| `sol.original.lambda` | Lagrange multipliers: one per variable, then one per constraint. |

[Solver behaviour](@ref) explains how SNOPT status codes map to return codes and lists
all fields of `sol.original`.

## Add bounds and constraints

Next, we solve problem 71 of the Hock–Schittkowski test set:

```math
\begin{aligned}
\min_{x} \quad & x_1 x_4 (x_1 + x_2 + x_3) + x_3 \\
\text{subject to} \quad & x_1 x_2 x_3 x_4 \ge 25, \\
& x_1^2 + x_2^2 + x_3^2 + x_4^2 = 40, \\
& 1 \le x_i \le 5, \quad i = 1, \dots, 4.
\end{aligned}
```

The optimal objective value is approximately ``17.014``.

The constraint function has the in-place signature `cons(res, x, p)` and writes one
value per constraint into `res`. The keywords `lb` and `ub` bound the variables, and
`lcons` and `ucons` bound the constraint values; equal lower and upper values give an
equality constraint:

```jldoctest hs71
using OptimizationSNOPT

objective(x, p) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]

function constraints!(res, x, p)
    res[1] = x[1] * x[2] * x[3] * x[4]
    res[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    return nothing
end

f = OptimizationFunction(objective, AutoForwardDiff(); cons = constraints!)
prob = OptimizationProblem(f, [1.0, 5.0, 5.0, 1.0];
                           lb = fill(1.0, 4), ub = fill(5.0, 4),
                           lcons = [25.0, 40.0], ucons = [Inf, 40.0])
sol = solve(prob, SnoptOptimizer())
(sol.retcode, round(sol.objective; digits = 3))

# output

(SciMLBase.ReturnCode.Success, 17.014)
```

`AutoForwardDiff()` now also computes the constraint Jacobian, the matrix
``J_{ij} = \partial c_i / \partial x_j``.

## Use a sparse Jacobian

SNOPT is designed for problems in which most Jacobian entries are zero. By default,
OptimizationSNOPT.jl passes the Jacobian to SNOPT as a dense matrix. For a large
problem, wrap the backend in `AutoSparse`. Optimization.jl then detects the sparsity
pattern, the positions of the entries that can be nonzero, and OptimizationSNOPT.jl
passes only those entries to SNOPT. No other package needs to be loaded:

```julia
f = OptimizationFunction(objective, AutoSparse(AutoForwardDiff()); cons = constraints!)
```

A hand-written Jacobian (the `cons_j` keyword) can give its pattern with the
`cons_jac_prototype` keyword of `OptimizationFunction`.

## Next steps

- Change tolerances and limits: [Set solver options](@ref).
- Print progress or stop on a custom condition: [Monitor and stop a solve](@ref).
- Solve the problem again from a new starting point: [Solve a problem several times](@ref).

The repository contains the constrained example, solved with `AutoForwardDiff`,
`AutoFiniteDiff` and `AutoSparse(AutoForwardDiff())`, in `examples/hs71.jl`. Run it
from the root of a clone of the repository:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/hs71.jl
```
