# Configuration

## Solver options

Pass common Optimization.jl controls to `solve`.

| Keyword | Meaning |
| --- | --- |
| `maxiters` | Maximum major iterations |
| `maxtime` | Maximum elapsed seconds |
| `abstol` | Major optimality tolerance |
| `reltol` | Minor feasibility tolerance |
| `callback` | Function called during the solve |
| `verbose` or `show_trace` | Print an iteration trace |
| `store_trace` | Store iteration records |

Pass SNOPT settings to [`SnoptOptimizer`](@ref).

```julia
optimizer = SnoptOptimizer(
    major_iterations_limit = 2_000,
    major_optimality_tolerance = 1.0e-8,
    hessian = "limited_memory",
    additional_options = Dict("Linesearch tolerance" => 0.9),
)

solution = solve(problem, optimizer; maxiters = 500)
```

Explicit optimizer fields cannot also appear in `additional_options`.
The constructor rejects duplicate or invalid settings.

## Callbacks

The Optimization.jl callback receives the current optimization state and loss.
Return `true` to stop the solve. Return `false` to continue.

```julia
callback = (state, loss) -> loss < 1.0e-10
solution = solve(problem, SnoptOptimizer(); callback)
```

## Traces

Use `verbose = true` to print a minimal trace. Store records with
`store_trace = Val(true)`.

```julia
solution = solve(
    problem,
    SnoptOptimizer();
    verbose = true,
    store_trace = Val(true),
    trace_level = SnoptTraceAll(print_frequency = 2, store_frequency = 1),
)

solution.trace
```

`SnoptTraceMinimal` stores scalar progress values. `SnoptTraceAll` also stores
the point and constraint vectors.

## Cached solves

Use `init` when several solves share one problem structure.

```julia
cache = init(problem, SnoptOptimizer())
first_solution = solve!(cache)
reinit!(cache; u0 = [0.8, 0.8])
second_solution = solve!(cache)
```

SNOPT owns one Fortran workspace per process. OptimizationSNOPT serializes all
workspace operations. Use separate Julia processes for parallel solves.
