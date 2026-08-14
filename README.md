# OptimizationSNOPT.jl

[![CI](https://github.com/EllissoideRotondo/OptimizationSNOPT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/EllissoideRotondo/OptimizationSNOPT.jl/actions/workflows/CI.yml)

[Optimization.jl](https://github.com/SciML/Optimization.jl) /
[SciML](https://sciml.ai) wrapper for [SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/),
the sparse SQP nonlinear optimizer, via the
[SNOPT.jl](https://github.com/EllissoideRotondo/SNOPT.jl) low-level interface.

SNOPT is a closed-source commercial solver; you must provide your own licensed
`libsnopt7` shared library. See the
[SNOPT.jl installation notes](https://github.com/EllissoideRotondo/SNOPT.jl#installation)
for how the library is located (`SNOPTDIR`, platform library path, or the
system loader's default paths).

## Usage

```julia
using OptimizationBase, OptimizationSNOPT

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
prob = OptimizationProblem(
    OptimizationFunction(rosenbrock, AutoForwardDiff()),
    zeros(2), [1.0, 100.0]
)
sol = solve(prob, SnoptOptimizer())
```

Constrained problems, variable bounds, `MaxSense`, the `init`/`solve!` cache
interface, and the common `solve` keyword arguments (`maxiters`, `maxtime`,
`abstol`, `reltol`, `verbose`, `callback`) are supported. SNOPT-specific
settings go through the `SnoptOptimizer` fields and its `additional_options`
dictionary; see the `SnoptOptimizer` docstring for the full list, and
[the SNOPT option reference](https://ccom.ucsd.edu/~optimizers/docs/snopt/options.html)
for option semantics.

```julia
opt = SnoptOptimizer(
    major_iterations_limit = 2000,
    major_optimality_tolerance = 1e-8,
    additional_options = Dict("Linesearch tolerance" => 0.9),
)
sol = solve(prob, opt; verbose = Val(true))
```

Iteration traces can be printed (`verbose`/`show_trace`) or stored
(`store_trace = Val(true)`, retrieved as `sol.original.trace`), with detail and
frequency controlled by `trace_level = SnoptTraceMinimal(...)` or
`SnoptTraceAll(...)`.

## Concurrency

SNOPT keeps global Fortran state: only one solve can run per process.
`OptimizationSNOPT` serializes concurrent `solve` calls with an internal lock,
so threaded callers are safe but will not see parallel speedup. Use multiple
Julia processes (e.g. `Distributed`) for parallel SNOPT solves.

## License

The wrapper is MIT licensed. The SNOPT solver itself is a commercial product
whose binaries are **not** distributed with this package.

## Testing

CI on GitHub-hosted runners has no SNOPT library, so it only exercises the
library-free code paths. The full solver suite runs locally against a licensed
`libsnopt7` (Linux and Windows).
