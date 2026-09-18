# OptimizationSNOPT.jl

OptimizationSNOPT.jl solves Optimization.jl problems with SNOPT.
SNOPT is a commercial solver for large, constrained nonlinear problems.

This package provides the Optimization.jl interface. The sibling
[SNOPT.jl](https://EllissoideRotondo.github.io/SNOPT.jl/dev/) package provides
direct access to SNOPT's native interfaces.

## Interfaces

| Need | Package |
| --- | --- |
| Optimization.jl problems and automatic derivatives | OptimizationSNOPT.jl |
| Direct `snOptA`, `snOptB`, or `snOptC` access | SNOPT.jl |

Both packages require your own licensed SNOPT shared library.

## Documentation

- [Installation](@ref) covers package and SNOPT library setup.
- [Getting started](@ref) provides unconstrained and constrained examples.
- [Configuration](@ref) covers options, callbacks, traces, and cached solves.
- [API reference](@ref api-reference) lists public types and functions.

## Supported Optimization.jl features

- Variable bounds and nonlinear constraints.
- Minimization and maximization.
- Automatic or supplied first derivatives.
- Callbacks and early termination.
- Cached `init` and `solve!` workflows.
- Printed and stored iteration traces.

SNOPT solves continuous problems. Integer variables are relaxed to continuous
variables, and the package emits a warning.
