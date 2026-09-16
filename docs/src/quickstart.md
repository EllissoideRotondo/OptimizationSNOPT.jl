# Getting started

This example minimizes the Rosenbrock function. Automatic differentiation
computes the required gradient.

```julia
using OptimizationSNOPT

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

problem = OptimizationProblem(
    OptimizationFunction(rosenbrock, AutoForwardDiff()),
    zeros(2),
    [1.0, 100.0],
)

solution = solve(problem, SnoptOptimizer())
```

The solution stores the final point, objective, and return code:

```julia
solution.u
solution.objective
solution.retcode
```

## Add bounds and constraints

`lb` and `ub` bound the variables. `lcons` and `ucons` bound the constraint
values.

```julia
objective(x, _) = (x[1] - 2.0)^2 + (x[2] - 1.0)^2

function constraints!(result, x, _)
    result[1] = x[1] + x[2]
    return nothing
end

optimization_function = OptimizationFunction(
    objective,
    AutoForwardDiff();
    cons = constraints!,
)

problem = OptimizationProblem(
    optimization_function,
    [0.5, 0.5],
    nothing;
    lb = [0.0, 0.0],
    ub = [3.0, 3.0],
    lcons = [1.0],
    ucons = [1.0],
)

solution = solve(problem, SnoptOptimizer())
```

The constraint callback mutates `result`. Its length must equal the number of
constraint bounds.

## Run the worked example

The repository example compares three derivative methods on a standard
constrained problem.

```bash
julia --project=. examples/hs71.jl
```
