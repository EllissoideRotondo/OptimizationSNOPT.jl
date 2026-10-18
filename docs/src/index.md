```@meta
CurrentModule = OptimizationSNOPT
```

# OptimizationSNOPT.jl: SNOPT for Optimization.jl

OptimizationSNOPT.jl connects [SNOPT](https://ccom.ucsd.edu/~optimizers/solvers/snopt/),
a commercial solver for large, sparse, constrained nonlinear optimization problems, to
[Optimization.jl](https://docs.sciml.ai/Optimization/stable/). A problem defined as an
`OptimizationProblem` is solved with SNOPT by passing [`SnoptOptimizer`](@ref) to
`solve`, and the derivatives that SNOPT needs come from any automatic differentiation
backend that Optimization.jl supports, so the objective and the constraints can stay
ordinary Julia functions.

OptimizationSNOPT.jl solves problems of the form

```math
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & f(x, p) \\
\text{subject to} \quad & l_x \le x \le u_x, \\
& l_c \le c(x, p) \le u_c,
\end{aligned}
```

where ``f`` is the objective, ``c`` is a vector of constraint functions, ``p`` holds
the parameters, and the bounds can be infinite. SNOPT uses a sequential quadratic
programming (SQP) method, so ``f`` and ``c`` must be smooth.

!!! note "SNOPT license required"
    OptimizationSNOPT.jl does not include SNOPT. A SNOPT license and a compatible
    `libsnopt7` shared library are available from the SNOPT distributors.

## When to use OptimizationSNOPT.jl

With OptimizationSNOPT.jl, the same problem can be solved with SNOPT and with the
other Optimization.jl solvers, and the derivatives come from automatic
differentiation. The package is built on
[SNOPT.jl](https://EllissoideRotondo.github.io/SNOPT.jl/dev/), which wraps the three
SNOPT interfaces `snOptA`, `snOptB` and `snOptC`; use SNOPT.jl directly for linear
constraints given separately, finite-difference derivatives estimated by SNOPT, or
warm and hot starts.

## Installation

Install the package with the Julia package manager:

```julia
using Pkg
Pkg.add("OptimizationSNOPT")
```

Then, before starting Julia, set `SNOPTDIR` to the directory of the SNOPT library:

```bash
export SNOPTDIR=/path/to/snopt/lib
```

[Installation](installation.md) gives the commands for Windows and the setup check.

## Contents

### Tutorial and installation

The tutorial solves a first problem with bounds and nonlinear constraints. The
installation page makes the SNOPT library available to Julia.

```@contents
Pages = [
    "getting_started.md",
    "installation.md",
]
Depth = 2
```

### How-to guides

Each guide shows how to do one task with OptimizationSNOPT.jl.

```@contents
Pages = [
    "howto/options.md",
    "howto/monitoring.md",
    "howto/cache.md",
]
Depth = 2
```

### Solver behaviour and reference

The solver behaviour page explains return codes, SNOPT results and limits. The API
reference lists every exported type.

```@contents
Pages = [
    "concepts.md",
    "api.md",
]
Depth = 2
```

## Contributing

- Report bugs and request features on the
  [issue tracker](https://github.com/EllissoideRotondo/OptimizationSNOPT.jl/issues).
- Before opening a pull request, run the test suite with
  `julia --project=. -e 'using Pkg; Pkg.test()'` from the repository root.
  The solver tests run only when Julia finds a licensed `libsnopt7`.
