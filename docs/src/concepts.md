```@meta
CurrentModule = OptimizationSNOPT
```

# Solver behaviour

This page describes how OptimizationSNOPT.jl translates between Optimization.jl and
SNOPT: how a SNOPT status code becomes a return code, which SNOPT results the solution
keeps, how derivatives and maximization are handled, and which limits follow from
SNOPT and from the single SNOPT workspace per process. The
[SNOPT.jl concepts page](https://EllissoideRotondo.github.io/SNOPT.jl/dev/concepts/)
explains the SNOPT terms used here, such as major iterations and the basis.

## Return codes

When a solve ends, SNOPT returns an integer status code, which the solution keeps in
`sol.original.inform`. OptimizationSNOPT.jl maps this code to the Optimization.jl
return code in `sol.retcode`, so that program logic can test the return code in the
same way for every Optimization.jl solver:

| SNOPT status code | `sol.retcode` | Meaning |
| --- | --- | --- |
| 1 | `ReturnCode.Success` | Optimal solution found. |
| 2–6 | `ReturnCode.Success` | Partial success: a feasible point only (2), less than the requested accuracy (3), or a result of SNOPT's elastic mode (5–6). Check `sol.original.inform`. |
| 11–16 | `ReturnCode.Infeasible` | The constraints cannot be satisfied. |
| 21–22 | `ReturnCode.Failure` | The objective decreases without limit. |
| 31–32 | `ReturnCode.MaxIters` | An iteration limit was reached. |
| 34 | `ReturnCode.MaxTime` | The CPU time limit (`maxtime`) was reached. |
| 71–74 | `ReturnCode.Terminated` | The callback returned `true`. |
| all others | `ReturnCode.Failure` | Numerical difficulties, wrong derivatives, insufficient memory, and other errors. |

Because several SNOPT outcomes share `ReturnCode.Success` and `ReturnCode.Failure`,
check `sol.original.inform` when the exact reason matters; the
[SNOPT 7.7 user guide](https://ccom.ucsd.edu/~optimizers/static/pdfs/sndoc7.pdf#page=110)
explains each code.

## SNOPT results in the solution

`sol.original` is a `NamedTuple` with the SNOPT results that Optimization.jl has no
field for:

| Field | Content |
| --- | --- |
| `inform` | SNOPT status code. |
| `lambda` | Lagrange multipliers: one per variable, then one per constraint. |
| `iterations` | Total number of minor iterations. |
| `major_itns` | Number of major iterations. |
| `num_inf` | Number of bounds and constraints that are still violated. |
| `sum_inf` | Sum of these violations. |
| `trace` | The stored [`SnoptTrace`](@ref), or `nothing`; also available as `sol.trace`. |

`sol.stats` holds the Optimization.jl statistics: the wall-clock time of the solve and
the number of objective and gradient evaluations. Its iteration count is not reliable in
this version; use `sol.original.major_itns`.

## Derivatives

SNOPT needs the gradient of the objective and the Jacobian of the constraints, but no
second derivatives. OptimizationSNOPT.jl takes both from the `OptimizationFunction`:
either from the automatic differentiation backend given as its second argument, such
as `AutoForwardDiff()`, or from the functions passed as its `grad` and `cons_j`
keywords. A solve without either source fails with an error that names the missing
function.

The constraint Jacobian is passed to SNOPT as a dense matrix unless the
`OptimizationFunction` has a sparsity pattern, either from a sparse backend such as
`AutoSparse(AutoForwardDiff())` or from the `cons_jac_prototype` keyword. All
constraints are treated as nonlinear.

## Maximization

SNOPT only minimizes. For a problem created with `sense = MaxSense`,
OptimizationSNOPT.jl minimizes the negative objective and changes the sign back, so
`sol.objective`, the callback and the trace all show the objective of the original
problem.

## Integer variables

SNOPT solves continuous problems only. If the `OptimizationProblem` marks variables as
integer with the `int` keyword, OptimizationSNOPT.jl shows a warning and solves the
continuous relaxation, in which those variables can take any real value.

## One solve at a time

The SNOPT library allows only one active session, called the workspace, in each Julia
process. OptimizationSNOPT.jl therefore runs solves one at a time: when several tasks
or threads call `solve` at the same time, each call waits until the previous call has
finished. To solve several problems in parallel, start several Julia processes, for
example with the `Distributed` standard library.

Constructing a [`SnoptOptimizer`](@ref) and each solve open and close a workspace, so
they close any SNOPT.jl workspace that is open in the same process.

## Limits

- Linear constraints cannot be given separately; they are treated as nonlinear.
- Warm and hot starts are not available; each solve starts from `u0` only.
- SNOPT's print and summary files are not written.

[SNOPT.jl](https://EllissoideRotondo.github.io/SNOPT.jl/dev/) gives access to linear
constraints, warm and hot starts, and the SNOPT output files.
