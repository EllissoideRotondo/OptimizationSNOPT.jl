mutable struct SnoptCache{
        T, F <: OptimizationFunction, RC, LB, UB, I, S,
        JT <: AbstractMatrix{T}, CB, O,
    } <: SciMLBase.AbstractOptimizationCache
    const f::F
    const n::Int
    const num_cons::Int
    const reinit_cache::RC
    const lb::LB
    const ub::UB
    const int::I
    const lcons::Vector{T}
    const ucons::Vector{T}
    const sense::S
    J::JT
    const callback::CB
    const progress::Bool
    f_calls::Int
    f_grad_calls::Int
    const iterations::Ref{Int}
    obj_expr::Union{Expr, Nothing}
    cons_expr::Union{Vector{Expr}, Nothing}
    const opt::O
    const solver_args::NamedTuple
end

function Base.getproperty(cache::SnoptCache, name::Symbol)
    if name in fieldnames(OptimizationBase.ReInitCache)
        return getfield(cache.reinit_cache, name)
    end
    return getfield(cache, name)
end
function Base.setproperty!(cache::SnoptCache, name::Symbol, x)
    if name in fieldnames(OptimizationBase.ReInitCache)
        return setfield!(cache.reinit_cache, name, x)
    end
    return setfield!(cache, name, x)
end

function SciMLBase.get_p(
        sol::SciMLBase.OptimizationSolution{
            T,
            N,
            uType,
            C,
        }
    ) where {T, N, uType, C <: SnoptCache}
    return sol.cache.p
end
function SciMLBase.get_observed(
        sol::SciMLBase.OptimizationSolution{
            T,
            N,
            uType,
            C,
        }
    ) where {T, N, uType, C <: SnoptCache}
    return sol.cache.f.observed
end
function SciMLBase.get_syms(
        sol::SciMLBase.OptimizationSolution{
            T,
            N,
            uType,
            C,
        }
    ) where {T, N, uType, C <: SnoptCache}
    return variable_symbols(sol.cache.f)
end
function SciMLBase.get_paramsyms(
        sol::SciMLBase.OptimizationSolution{
            T,
            N,
            uType,
            C,
        }
    ) where {T, N, uType, C <: SnoptCache}
    return parameter_symbols(sol.cache.f)
end

function Base.getproperty(
        sol::SciMLBase.OptimizationSolution{T, N, uType, C, A, OV, O, ST},
        name::Symbol
    ) where {T, N, uType, C <: SnoptCache, A, OV, O, ST}
    if name === :trace
        original = getfield(sol, :original)
        if original isa NamedTuple && haskey(original, :trace)
            return original.trace
        end
        return nothing
    end
    return getfield(sol, name)
end

function SnoptCache(
        prob, opt;
        callback = nothing,
        progress = false,
        verbose = OptimizationBase.DEFAULT_VERBOSE,
        trace_level = SnoptTraceMinimal(),
        store_trace = Val(false),
        kwargs...
    )
    reinit_cache = OptimizationBase.ReInitCache(prob.u0, prob.p) # everything that can be changed via `reinit`
    show_trace = snopt_show_trace(verbose)
    final_verbose = verbose

    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    if prob.f.adtype isa ADTypes.AutoSymbolics || (
            prob.f.adtype isa ADTypes.AutoSparse &&
                prob.f.adtype.dense_ad isa ADTypes.AutoSymbolics
        )
        f = OptimizationBase.instantiate_function(
            prob.f, reinit_cache, prob.f.adtype, num_cons;
            g = true, cons_j = true
        )
    else
        f = OptimizationBase.instantiate_function(
            prob.f, reinit_cache, prob.f.adtype, num_cons;
            g = true, cons_j = true
        )
    end
    T = eltype(prob.u0)
    n = length(prob.u0)

    J = if isnothing(f.cons_jac_prototype)
        zeros(T, num_cons, n)
    else
        similar(f.cons_jac_prototype, T)
    end
    lcons = prob.lcons === nothing ? fill(T(-Inf), num_cons) : prob.lcons
    ucons = prob.ucons === nothing ? fill(T(Inf), num_cons) : prob.ucons

    sys = f.sys isa SymbolicIndexingInterface.SymbolCache{Nothing, Nothing, Nothing} ?
        nothing : f.sys
    obj_expr = f.expr
    cons_expr = f.cons_expr

    solver_args = merge(
        NamedTuple(kwargs),
        (; verbose = final_verbose, show_trace,
            trace_level = normalize_trace_level(trace_level), store_trace)
    )

    return SnoptCache(
        f,
        n,
        num_cons,
        reinit_cache,
        prob.lb,
        prob.ub,
        prob.int,
        lcons,
        ucons,
        prob.sense,
        J,
        callback,
        progress,
        0,
        0,
        Ref(0),
        obj_expr,
        cons_expr,
        opt,
        solver_args
    )
end

function eval_objective(cache::SnoptCache, x)
    l = cache.f(x, cache.p)
    cache.f_calls += 1
    return cache.sense === OptimizationBase.MaxSense ? -l : l
end

function eval_constraint(cache::SnoptCache, g, x)
    cache.f.cons(g, x)
    return
end

function eval_objective_gradient(cache::SnoptCache, G, x)
    if cache.f.grad === nothing
        error(
            "Use OptimizationFunction to pass the objective gradient or " *
                "automatically generate it with one of the autodiff backends." *
                "If you are using the ModelingToolkit symbolic interface, pass the `grad` kwarg set to `true` in `OptimizationProblem`."
        )
    end
    cache.f.grad(G, x)
    cache.f_grad_calls += 1

    if cache.sense === OptimizationBase.MaxSense
        G .*= -one(eltype(G))
    end

    return
end

function jacobian_structure(cache::SnoptCache)
    if cache.J isa SparseMatrixCSC
        rows, cols, _ = findnz(cache.J)
        inds = Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols)]
    else
        rows, cols = size(cache.J)
        inds = Tuple{Int, Int}[(i, j) for j in 1:cols for i in 1:rows]
    end
    return inds
end

function eval_constraint_jacobian(cache::SnoptCache, j, x)
    if isempty(j)
        return
    elseif cache.f.cons_j === nothing
        error(
            "Use OptimizationFunction to pass the constraints' jacobian or " *
                "automatically generate i with one of the autodiff backends." *
                "If you are using the ModelingToolkit symbolic interface, pass the `cons_j` kwarg set to `true` in `OptimizationProblem`."
        )
    end
    # Get and cache the Jacobian object here once. `evaluator.J` calls
    # `getproperty`, which is expensive because it calls `fieldnames`.
    J = cache.J
    cache.f.cons_j(J, x)
    if J isa SparseMatrixCSC
        nnz = nonzeros(J)
        length(j) == length(nnz) ||
            throw(DimensionMismatch(
                "SNOPT requested $(length(j)) Jacobian nonzeros, but the cached sparse Jacobian has $(length(nnz))"))
        for (i, Ji) in zip(eachindex(j), nnz)
            j[i] = Ji
        end
    else
        j .= vec(J)
    end
    return
end

