```@meta
CurrentModule = OptimizationSNOPT
```

# Set solver options

Solver settings come from two places. The common Optimization.jl keywords of `solve`,
such as `maxiters`, apply to the call. The fields of [`SnoptOptimizer`](@ref) hold
SNOPT settings, including any option from the SNOPT option reference. When both set
the same SNOPT option, the `solve` keyword wins.

## Common solve keywords

```julia
sol = solve(prob, SnoptOptimizer(); maxiters = 500, abstol = 1.0e-8)
```

| Keyword | SNOPT option | Meaning |
| --- | --- | --- |
| `maxiters` | `Major iterations limit` | Maximum number of major iterations. |
| `maxtime` | `Time limit` | Maximum CPU time of the solve, in seconds. |
| `abstol` | `Major optimality tolerance` | Tolerance on the optimality conditions. |
| `reltol` | `Minor feasibility tolerance` | Tolerance on the constraint violation in each QP subproblem. Not a relative tolerance. |

The `solve` keywords are applied after the optimizer fields and `additional_options`,
so `reltol`, for example, replaces `minor_feasibility_tolerance`. The keywords
`callback`, `verbose`, `trace_level` and `store_trace` are described in
[Monitor and stop a solve](@ref).

## Optimizer fields

```julia
opt = SnoptOptimizer(major_iterations_limit = 2_000,
                     major_optimality_tolerance = 1.0e-8,
                     hessian = "limited_memory")
sol = solve(prob, opt)
```

| Field | Default | SNOPT option |
| --- | --- | --- |
| `major_iterations_limit` | `1000` | `Major iterations limit` |
| `minor_iterations_limit` | `500` | `Minor iterations limit` |
| `major_optimality_tolerance` | `1.0e-6` | `Major optimality tolerance` |
| `major_feasibility_tolerance` | `1.0e-6` | `Major feasibility tolerance` |
| `minor_feasibility_tolerance` | `1.0e-6` | `Minor feasibility tolerance` |
| `hessian` | `nothing` | None, `Hessian full memory` or `Hessian limited memory` |
| `derivative_option` | `1` | `Derivative option`; keep the default |
| `major_print_level`, `minor_print_level` | `1`, `0` | `Major print level`, `Minor print level` |
| `additional_options` | empty | Any other option; see below |

SNOPT approximates the Hessian of the Lagrangian from the change in the gradients
between iterations, either as a dense matrix (full memory) or from a few recent
updates (limited memory). With the default `hessian = nothing`, no Hessian option is
passed and SNOPT applies its own rule: full memory when the problem has at most 75
nonlinear variables, limited memory otherwise. The dense update costs time and storage
that grow with the square of the number of nonlinear variables, so set
`hessian = "full_memory"` on a larger problem only when the limited-memory
approximation needs many more major iterations. Set `hessian = "limited_memory"` to
use the limited-memory update at any size.

OptimizationSNOPT.jl does not write SNOPT's print file or summary file, so the print
levels have no visible effect. To see the progress of a solve, use `verbose = true`
(see [Monitor and stop a solve](@ref)).

## Other SNOPT options

Pass any other SNOPT option in `additional_options`, as a dictionary from option name
to value:

```julia
opt = SnoptOptimizer(additional_options = Dict("Linesearch tolerance" => 0.9,
                                               :scale_option => 1))
```

| Part | Allowed values |
| --- | --- |
| Name | A `String` or a `Symbol`. `_` stands for a space, and case does not matter. |
| Value | An integer, a finite floating-point number, a `String` or `Symbol` (`_` stands for a space), or `nothing` for an option without a value. |

The [SNOPT option reference](https://ccom.ucsd.edu/~optimizers/docs/snopt/options.html)
gives all names, default values and meanings. An option that already has a field in
[`SnoptOptimizer`](@ref) cannot appear in `additional_options`.

## Errors in options

[`SnoptOptimizer`](@ref) checks its fields when it is constructed and throws an
`ArgumentError` for a value of the wrong type or range, for `true` or `false` as an
option value, for an option in `additional_options` that has its own field, or for an
option name that appears twice. If the SNOPT library is
available, the constructor also passes every option to SNOPT in a temporary
workspace, so an option that SNOPT does not accept fails at construction, not during
the solve.

The SNOPT library allows one active session, the workspace, per Julia process. The
constructor and each solve open a workspace, so they close any SNOPT.jl workspace
that is open in the same process; see [One solve at a time](@ref).
