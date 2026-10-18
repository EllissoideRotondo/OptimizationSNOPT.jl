```@meta
CurrentModule = OptimizationSNOPT
```

# API reference

OptimizationSNOPT.jl re-exports `OptimizationBase`, so `OptimizationProblem`,
`OptimizationFunction`, `solve`, `init`, `solve!`, `reinit!` and the automatic
differentiation selectors are available after `using OptimizationSNOPT`. The
[Optimization.jl documentation](https://docs.sciml.ai/Optimization/stable/) describes
them.

## Optimizer

```@docs
SnoptOptimizer
```

## Traces

```@docs
SnoptTraceMinimal
SnoptTraceAll
SnoptTrace
SnoptTraceEntry
```
