```@meta
CurrentModule = OptimizationSNOPT
```

# Monitor and stop a solve

A solve can print a table of its progress, store the same values for later use, and
call a function that stops the solve on a custom condition. By default, a solve
prints nothing.

## Print progress

Pass `verbose = true` to print one row per major iteration:

```julia
sol = solve(prob, SnoptOptimizer(); verbose = true)
```

The table starts with the main optimizer settings. Each row has these columns:

| Column | Content |
| --- | --- |
| `Major`, `Minor` | Number of the major iteration, and number of minor iterations in it. |
| `Objective` | Objective value. |
| `Constr viol` | Constraint violation, as SNOPT measures it after scaling the problem. |
| `Optimality` | Error in the optimality conditions, as SNOPT measures it. |
| `Step` | Length of the last step. |

`show_trace = true` is the same as `verbose = true`. The
[SNOPT.jl concepts page](https://EllissoideRotondo.github.io/SNOPT.jl/dev/concepts/)
explains major and minor iterations.

## Store a trace

Pass `store_trace = true` (or `Val(true)`) to keep the rows in the solution:

```julia
sol = solve(prob, SnoptOptimizer(); store_trace = true)
trace = sol.trace                 # a SnoptTrace
last_row = trace.history[end]     # a SnoptTraceEntry
```

`trace_level` selects how much each row contains and how often a row is printed or
stored:

| `trace_level` | Stored row | Printed table |
| --- | --- | --- |
| [`SnoptTraceMinimal`](@ref)`()` (default) | Scalar values, including the largest bound violation; `x` and `c` are `nothing`. | The columns above. |
| [`SnoptTraceAll`](@ref)`()` | Scalar values and copies of the point `x` and the constraint values `c`. | The columns above and the largest bound violation. |

```julia
sol = solve(prob, SnoptOptimizer();
            verbose = true, store_trace = true,
            trace_level = SnoptTraceAll(print_frequency = 10, store_frequency = 1))
```

With these settings, the solve prints every tenth major iteration and stores every
major iteration. The last row of a trace has `iteration == -1` and holds the final
values and the SNOPT status code. Without `store_trace`, `sol.trace` is `nothing`.

## Stop on a custom condition

Pass an Optimization.jl callback to `solve`. The callback receives an
`OptimizationState` and the current objective value; it returns `true` to stop the
solve and `false` to continue:

```julia
deadline = time() + 30.0
callback = (state, loss) -> time() > deadline
sol = solve(prob, SnoptOptimizer(); callback = callback)
```

This example stops the solve after 30 seconds of wall-clock time. After a stop, the
return code is `ReturnCode.Terminated`.

| `state` field | Content |
| --- | --- |
| `state.iter` | Always `0` in this version; the trace gives iteration numbers. |
| `state.u` | Point of the latest evaluation, which can be a trial point of the line search. |
| `state.objective` | Current objective value, the same as `loss`. |

The callback is called after every evaluation of the objective, which happens several
times per major iteration. `state.u` is reused between calls, so copy it to keep its
values. To limit only the number of iterations or the CPU time, use the `solve`
keywords `maxiters` and `maxtime` instead (see [Set solver options](@ref)).
