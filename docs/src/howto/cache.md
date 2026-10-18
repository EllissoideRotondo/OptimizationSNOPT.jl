```@meta
CurrentModule = OptimizationSNOPT
```

# Solve a problem several times

To solve the same problem from several starting points, or with several parameter
values, create a cache once with `init` and solve it with `solve!`. The cache keeps
the derivative functions that Optimization.jl prepared for the problem, so later
solves do not prepare them again.

```julia
cache = init(prob, SnoptOptimizer())
sol1 = solve!(cache)

reinit!(cache; u0 = [0.8, 0.8])        # new starting point
sol2 = solve!(cache)

reinit!(cache; p = [1.0, 50.0])        # new parameter values
sol3 = solve!(cache)
```

`reinit!` changes only the starting point `u0` and the parameters `p`. The keywords
passed to `init`, such as `maxiters` or `callback`, apply to every `solve!` of the
cache. To change the bounds, the constraints or the optimizer, create a new problem
or a new cache.

Each `solve!` is a new SNOPT solve that starts from `u0`. SNOPT does not reuse its
basis (the record of which variables are at a bound) or its Hessian approximation
from the previous solve; for warm and hot starts,
use [SNOPT.jl](https://EllissoideRotondo.github.io/SNOPT.jl/dev/howto/warm_start/)
directly.
