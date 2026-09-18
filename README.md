# OptimizationSNOPT.jl

[![CI](https://github.com/EllissoideRotondo/OptimizationSNOPT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/EllissoideRotondo/OptimizationSNOPT.jl/actions/workflows/CI.yml)
[![docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://EllissoideRotondo.github.io/OptimizationSNOPT.jl/dev/)

OptimizationSNOPT.jl connects [Optimization.jl](https://github.com/SciML/Optimization.jl)
problems to [SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/).
SNOPT solves large, constrained nonlinear optimization problems.

Use [SNOPT.jl](https://github.com/EllissoideRotondo/SNOPT.jl) for direct access
to SNOPT's native problem interfaces. Use this package for Optimization.jl
problems and automatic differentiation.

## Requirements

- Julia 1.10 or later.
- A licensed SNOPT 7 shared library that includes the C API provided by
  [`snopt-interface`](https://github.com/snopt/snopt-interface).

You must obtain the SNOPT library and license separately.

## Installation

Install OptimizationSNOPT.jl with Julia's package manager:

```julia
import Pkg
Pkg.add("OptimizationSNOPT")
```

For a source checkout, place both repositories in the same directory:

```bash
git clone https://github.com/EllissoideRotondo/SNOPT.jl SNOPT
git clone https://github.com/EllissoideRotondo/OptimizationSNOPT.jl OptimizationSNOPT
julia --project=OptimizationSNOPT -e 'using Pkg; Pkg.develop(path="SNOPT"); Pkg.instantiate()'
```

Set `SNOPTDIR` to the directory that contains the shared library:

```bash
export SNOPTDIR=/path/to/snopt/lib
```

Windows PowerShell uses this equivalent command:

```powershell
$env:SNOPTDIR = "C:\path\to\snopt\lib"
```

From the OptimizationSNOPT.jl repository, verify library discovery:

```bash
julia --project=. -e 'using SNOPT; @assert SNOPT.has_snopt(); println("SNOPT is ready")'
```

The [SNOPT.jl installation guide](https://EllissoideRotondo.github.io/SNOPT.jl/dev/installation/)
lists library names, search paths, licensing, and platform limits.

## Getting started

This example minimizes the Rosenbrock function with automatic differentiation:

```julia
using OptimizationSNOPT

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

problem = OptimizationProblem(
    OptimizationFunction(rosenbrock, AutoForwardDiff()),
    zeros(2),
    [1.0, 100.0],
)

solution = solve(problem, SnoptOptimizer())
solution.u
solution.objective
solution.retcode
```

Run the constrained worked example from the repository root:

```bash
julia --project=. examples/hs71.jl
```

## Solver configuration

Pass common Optimization.jl keywords to `solve`. Pass SNOPT options to
`SnoptOptimizer`.

```julia
optimizer = SnoptOptimizer(
    major_iterations_limit = 2_000,
    major_optimality_tolerance = 1.0e-8,
    additional_options = Dict("Linesearch tolerance" => 0.9),
)

solution = solve(problem, optimizer; maxiters = 500, verbose = true)
```

OptimizationSNOPT supports bounds, nonlinear constraints, maximization, callbacks,
cached solves, and iteration traces. See the
[`SnoptOptimizer` API](https://EllissoideRotondo.github.io/OptimizationSNOPT.jl/dev/api/)
for accepted fields and common solve keywords.

## Concurrency

SNOPT owns one global Fortran workspace per process. This package serializes
all solves with a lock. Threaded solves are safe, but they run sequentially.

Use separate Julia processes for parallel solves.

## Testing

Run the full suite from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Solver tests run when Julia finds `libsnopt7`. Other tests remain available
without the library.

## License

OptimizationSNOPT.jl uses the MIT License. SNOPT is a separate commercial
product. Its binaries are not distributed with this package.
