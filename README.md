# OptimizationSNOPT.jl

[![CI](https://github.com/EllissoideRotondo/OptimizationSNOPT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/EllissoideRotondo/OptimizationSNOPT.jl/actions/workflows/CI.yml)
[![docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://EllissoideRotondo.github.io/OptimizationSNOPT.jl/dev/)

OptimizationSNOPT.jl connects [SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/),
a commercial solver for large, sparse, constrained nonlinear optimization problems, to
[Optimization.jl](https://docs.sciml.ai/Optimization/stable/). A problem defined as an
`OptimizationProblem` is solved with SNOPT by passing `SnoptOptimizer()` to `solve`,
and the derivatives come from any automatic differentiation backend that
Optimization.jl supports.

With OptimizationSNOPT.jl, the same problem can be solved with SNOPT and with the
other Optimization.jl solvers. The package is built on
[SNOPT.jl](https://github.com/EllissoideRotondo/SNOPT.jl), which wraps the three SNOPT
interfaces `snOptA`, `snOptB` and `snOptC`; use SNOPT.jl directly for linear
constraints given separately, finite-difference derivatives estimated by SNOPT, or warm
and hot starts.

## Installation

OptimizationSNOPT.jl needs Julia 1.11 or later and a licensed SNOPT 7.7 library,
`libsnopt7`, that includes the [snopt-interface](https://github.com/snopt/snopt-interface)
C functions. OptimizationSNOPT.jl does not include SNOPT; the library and the license
are available from the SNOPT distributors.

```julia
using Pkg
Pkg.add("OptimizationSNOPT")
```

Before starting Julia, set `SNOPTDIR` to the directory that contains the library:

```bash
export SNOPTDIR=/path/to/snopt/lib
```

The [installation guide](https://EllissoideRotondo.github.io/OptimizationSNOPT.jl/dev/installation/)
gives the Windows commands and the setup check.

## Example

Minimize the Rosenbrock function, with the gradient from ForwardDiff.jl:

```julia-repl
julia> using OptimizationSNOPT

julia> rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2;

julia> f = OptimizationFunction(rosenbrock, AutoForwardDiff());

julia> prob = OptimizationProblem(f, [0.0, 0.0], [1.0, 100.0]);

julia> sol = solve(prob, SnoptOptimizer());

julia> sol.retcode
ReturnCode.Success = 1

julia> round.(sol.u; digits = 6)
2-element Vector{Float64}:
 1.0
 1.0
```

For bounds and nonlinear constraints, pass `lb`, `ub`, `lcons` and `ucons` to
`OptimizationProblem` and `cons` to `OptimizationFunction`, as the
[getting started tutorial](https://EllissoideRotondo.github.io/OptimizationSNOPT.jl/dev/getting_started/)
shows.

## Documentation

The [documentation](https://EllissoideRotondo.github.io/OptimizationSNOPT.jl/dev/)
starts with a
[tutorial](https://EllissoideRotondo.github.io/OptimizationSNOPT.jl/dev/getting_started/)
on bounds, nonlinear constraints and sparse Jacobians. It then gives how-to guides for
solver options, monitoring and repeated solves, explains how SNOPT results map to
Optimization.jl, and ends with the API reference.

## Limits

- One Julia process runs one SNOPT solve at a time. Use several processes for
  parallel solves.
- The constraint Jacobian is dense unless the backend is sparse, for example
  `AutoSparse(AutoForwardDiff())`, or `cons_jac_prototype` gives its pattern.
- Integer variables are relaxed to continuous variables, with a warning.

[Solver behaviour](https://EllissoideRotondo.github.io/OptimizationSNOPT.jl/dev/concepts/#Limits)
lists all limits.

## License

OptimizationSNOPT.jl is released under the MIT License. SNOPT is a separate commercial
product with its own license, and its binaries are not distributed with this package.
