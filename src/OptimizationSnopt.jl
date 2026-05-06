module OptimizationSNOPT

using Reexport
@reexport using OptimizationBase
using SNOPT
using LinearAlgebra
using Printf
using SparseArrays
using SciMLBase
using SymbolicIndexingInterface

export SnoptOptimizer
export SnoptTrace
export SnoptTraceAll
export SnoptTraceEntry
export SnoptTraceMinimal

"""
    SnoptOptimizer(; kwargs...)

Optimizer using SNOPT (Sparse Nonlinear OPTimizer) for nonlinear optimization.

SNOPT solves problems of the form:

    min  f(x)
    s.t. g_L ≤ g(x) ≤ g_U
         x_L ≤  x   ≤ x_U

# Common Interface Arguments

The following common optimization arguments can be passed to `solve`:
- `maxiters`: Overrides the `major_iterations_limit` option
- `abstol`: Overrides the `major_optimality_tolerance` option
- `reltol`: Overrides the `minor_feasibility_tolerance` option
- `verbose`: When `true` or `Val(true)`, prints an OptimizationSnopt trace
- `show_trace`: Alias for `verbose`, following the SciML diagnostics API
- `trace_level`: `SnoptTraceMinimal()` or `SnoptTraceAll()`; supports print/store frequency
- `store_trace`: When `Val(true)`, stores the trace in `sol.original.trace`

# Keyword Arguments

## Output Options
- `major_print_level::Int = 1`: Print level for major iterations (0 = silent, 1 = summary)
- `minor_print_level::Int = 0`: Print level for minor iterations

## Iteration Limits
- `major_iterations_limit::Int = 1000`: Maximum number of major iterations
- `minor_iterations_limit::Int = 500`: Maximum number of minor iterations

## Tolerances
- `major_optimality_tolerance::Float64 = 1e-6`: KKT optimality tolerance
- `major_feasibility_tolerance::Float64 = 1e-6`: Nonlinear constraint feasibility tolerance
- `minor_feasibility_tolerance::Float64 = 1e-6`: Linear feasibility tolerance for QP subproblems

## Derivative Options
- `derivative_option::Int = 1`: Derivative mode (1 = gradient only, 3 = gradient + Jacobian)

## Additional Options
- `additional_options::AbstractDict`: Any other SNOPT option not explicitly listed above,
  using strings or symbols as keys (spaces or underscores allowed, case-insensitive). Values
  may be integers, floats, strings/symbols for option words, or `nothing` for bare options.
  Keys and values are validated by SNOPT at construction time when `libsnopt7` is available.

# Examples

```julia
using OptimizationBase, OptimizationSnopt

opt = SnoptOptimizer()

opt = SnoptOptimizer(
    major_iterations_limit = 2000,
    major_optimality_tolerance = 1e-8,
    additional_options = Dict("Linesearch tolerance" => 0.9)
)

result = solve(prob, opt; maxiters = 500, abstol = 1e-8, verbose = Val(true))
```

# References

For complete documentation of all Snopt options, see:
https://ccom.ucsd.edu/~optimizers/docs/snopt/options.html
"""
@kwdef struct SnoptOptimizer <: SciMLBase.AbstractOptimizationAlgorithm
    # Output
    major_print_level::Int = 1
    minor_print_level::Int = 0

    # Iteration limits
    major_iterations_limit::Int = 1000
    minor_iterations_limit::Int = 500

    # Tolerances
    major_optimality_tolerance::Float64 = 1.0e-6
    major_feasibility_tolerance::Float64 = 1.0e-6
    minor_feasibility_tolerance::Float64 = 1.0e-6

    # Derivative handling
    derivative_option::Int = 1

    # Hessian approximation: "full_memory" or "limited_memory"
    hessian::String = "full_memory"

    # Catch-all for any other SNOPT option string
    additional_options::Dict{String, Any} = Dict{String, Any}()

    function SnoptOptimizer(
            major_print_level, minor_print_level,
            major_iterations_limit, minor_iterations_limit,
            major_optimality_tolerance, major_feasibility_tolerance, minor_feasibility_tolerance,
            derivative_option, hessian, additional_options)
        major_print_level = validate_nonnegative_int(:major_print_level, major_print_level)
        minor_print_level = validate_nonnegative_int(:minor_print_level, minor_print_level)
        major_iterations_limit = validate_positive_int(:major_iterations_limit, major_iterations_limit)
        minor_iterations_limit = validate_positive_int(:minor_iterations_limit, minor_iterations_limit)
        major_optimality_tolerance = validate_positive_float(
            :major_optimality_tolerance, major_optimality_tolerance)
        major_feasibility_tolerance = validate_positive_float(
            :major_feasibility_tolerance, major_feasibility_tolerance)
        minor_feasibility_tolerance = validate_positive_float(
            :minor_feasibility_tolerance, minor_feasibility_tolerance)
        derivative_option = validate_derivative_option(derivative_option)
        hessian = validate_hessian_option(hessian)
        additional_options = validate_additional_options(additional_options)
        validate_snopt_optimizer_options!(
            major_print_level,
            minor_print_level,
            major_iterations_limit,
            minor_iterations_limit,
            major_optimality_tolerance,
            major_feasibility_tolerance,
            minor_feasibility_tolerance,
            derivative_option,
            hessian,
            additional_options
        )
        new(major_print_level, minor_print_level,
            major_iterations_limit, minor_iterations_limit,
            major_optimality_tolerance, major_feasibility_tolerance, minor_feasibility_tolerance,
            derivative_option, hessian, additional_options)
    end
end

function SciMLBase.has_init(::SnoptOptimizer)
    return true
end

SciMLBase.allowscallback(alg::SnoptOptimizer) = true
OptimizationBase.supports_sense(::SnoptOptimizer) = true

SciMLBase.supports_opt_cache_interface(alg::SnoptOptimizer) = true
SciMLBase.requiresgradient(opt::SnoptOptimizer) = true
SciMLBase.requireshessian(opt::SnoptOptimizer) = false
SciMLBase.requiresconsjac(opt::SnoptOptimizer) = true
SciMLBase.requiresconshess(opt::SnoptOptimizer) = false
SciMLBase.allowsbounds(opt::SnoptOptimizer) = true
SciMLBase.allowsconstraints(opt::SnoptOptimizer) = true

include("callback.jl")
include("cache.jl")

snopt_show_trace(verbose) = false
snopt_show_trace(verbose::Bool) = verbose
snopt_show_trace(::Val{true}) = true
snopt_show_trace(::Val{false}) = false
snopt_show_trace(::SciMLLogging.None) = false
snopt_show_trace(::SciMLLogging.AbstractVerbosityPreset) = true
snopt_log_trace_available() = isdefined(Snopt, :SnoptMajorLog)

function validate_integer_option(name::Symbol, value)
    value isa Integer && !(value isa Bool) ||
        throw(ArgumentError("$(name) must be an integer, got $(repr(value))"))
    return Int(value)
end

function validate_nonnegative_int(name::Symbol, value)
    value = validate_integer_option(name, value)
    value >= 0 ||
        throw(ArgumentError("$(name) must be ≥ 0, got $value"))
    return value
end

function validate_positive_int(name::Symbol, value)
    value = validate_integer_option(name, value)
    value > 0 ||
        throw(ArgumentError("$(name) must be > 0, got $value"))
    return value
end

function validate_positive_float(name::Symbol, value)
    value isa Real && !(value isa Bool) ||
        throw(ArgumentError("$(name) must be a real number, got $(repr(value))"))
    value = Float64(value)
    isfinite(value) ||
        throw(ArgumentError("$(name) must be finite, got $value"))
    value > 0 ||
        throw(ArgumentError("$(name) must be > 0, got $value"))
    return value
end

function validate_derivative_option(value)
    value = validate_integer_option(:derivative_option, value)
    value in (1, 2, 3) ||
        throw(ArgumentError("derivative_option must be 1, 2, or 3, got $value"))
    return value
end

function validate_hessian_option(value)
    value isa AbstractString ||
        throw(ArgumentError("hessian must be \"full_memory\" or \"limited_memory\", got $(repr(value))"))
    value = String(value)
    value in ("full_memory", "limited_memory") ||
        throw(ArgumentError("hessian must be \"full_memory\" or \"limited_memory\", got $(repr(value))"))
    return value
end

function normalize_snopt_option_key(key)
    key isa Union{AbstractString, Symbol} ||
        throw(ArgumentError("SNOPT option keys must be strings or symbols, got $(repr(key))"))
    normalized = replace(String(key), '_' => ' ')
    normalized = strip(normalized)
    isempty(normalized) &&
        throw(ArgumentError("SNOPT option keys must not be empty"))
    return uppercasefirst(normalized)
end

function normalize_snopt_option_value(key::String, value)
    if value === nothing
        return nothing
    elseif value isa Bool
        throw(ArgumentError(
            "SNOPT option $(repr(key)) received Bool value $(repr(value)); " *
            "use an integer, float, string, or `nothing` for bare options"))
    elseif value isa Integer
        return Int(value)
    elseif value isa AbstractFloat
        value = Float64(value)
        isfinite(value) ||
            throw(ArgumentError("SNOPT option $(repr(key)) must be finite, got $value"))
        return value
    elseif value isa AbstractString || value isa Symbol
        normalized = replace(String(value), '_' => ' ')
        normalized = String(strip(normalized))
        isempty(normalized) &&
            throw(ArgumentError("SNOPT option $(repr(key)) received an empty string value"))
        return normalized
    end
    throw(ArgumentError(
        "SNOPT option $(repr(key)) has unsupported value $(repr(value)); " *
        "use an integer, float, string, symbol, or `nothing`"))
end

function validate_additional_options(options)
    options isa AbstractDict ||
        throw(ArgumentError("additional_options must be an AbstractDict, got $(typeof(options))"))
    normalized = Dict{String, Any}()
    for (key, value) in pairs(options)
        normalized_key = normalize_snopt_option_key(key)
        haskey(normalized, normalized_key) &&
            throw(ArgumentError("duplicate SNOPT option key after normalization: $(repr(normalized_key))"))
        normalized[normalized_key] = normalize_snopt_option_value(normalized_key, value)
    end
    return normalized
end

function hessian_option_string(hessian::String)
    return replace(hessian, "_" => " ")
end

const SNOPT_BOUND_INF = 1.0e20

function snopt_bound_value(value)
    value = Float64(value)
    value == Inf && return SNOPT_BOUND_INF
    value == -Inf && return -SNOPT_BOUND_INF
    return value
end

snopt_bound_vector(values) = snopt_bound_value.(collect(values))

function snopt_optimizer_option_pairs(
        major_print_level,
        minor_print_level,
        major_iterations_limit,
        minor_iterations_limit,
        major_optimality_tolerance,
        major_feasibility_tolerance,
        minor_feasibility_tolerance,
        derivative_option,
        hessian,
        additional_options;
        include_print_levels::Bool = true
    )
    options = Tuple{String, Any}[]
    if include_print_levels
        push!(options, ("Major print level", major_print_level))
        push!(options, ("Minor print level", minor_print_level))
    end
    push!(options, ("Major iterations limit", major_iterations_limit))
    push!(options, ("Minor iterations limit", minor_iterations_limit))
    push!(options, ("Major optimality tolerance", major_optimality_tolerance))
    push!(options, ("Major feasibility tolerance", major_feasibility_tolerance))
    push!(options, ("Minor feasibility tolerance", minor_feasibility_tolerance))
    push!(options, ("Derivative option", derivative_option))
    push!(options, ("Hessian $(hessian_option_string(hessian))", nothing))
    for (key, value) in additional_options
        push!(options, (key, value))
    end
    return options
end

snopt_optimizer_option_pairs(opt::SnoptOptimizer; include_print_levels::Bool = true) =
    snopt_optimizer_option_pairs(
        opt.major_print_level,
        opt.minor_print_level,
        opt.major_iterations_limit,
        opt.minor_iterations_limit,
        opt.major_optimality_tolerance,
        opt.major_feasibility_tolerance,
        opt.minor_feasibility_tolerance,
        opt.derivative_option,
        opt.hessian,
        opt.additional_options;
        include_print_levels
    )

function snopt_option_label(key::String, value)
    value === nothing && return repr(key)
    return "$(repr(key)) => $(repr(value))"
end

function apply_snopt_option!(ws, key::String, value)
    errors = nothing
    if value === nothing
        errors = set_option!(ws, key)
    elseif value isa Int
        errors = set_option!(ws, key, value)
    elseif value isa Float64
        errors = set_option!(ws, key, value)
    elseif value isa String
        errors = set_option!(ws, string(key, " ", value))
    else
        throw(ArgumentError("unsupported normalized SNOPT option value $(repr(value))"))
    end
    if errors isa Integer && errors != 0
        msg = value === nothing ?
            "SNOPT: Couldn't set option '$(key)'" :
            "SNOPT: Couldn't set option '$(key)' to value '$(value)'"
        throw(ArgumentError(msg))
    end
    return ws
end

function validate_snopt_option_pairs_with_library!(options)
    Snopt.has_snopt() || return nothing
    ws = initialize("", "", 1000, 1000)
    try
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                for (key, value) in options
                    apply_snopt_option!(ws, key, value)
                end
            end
        end
    finally
        finalize(ws)
    end
    return nothing
end

function validate_snopt_optimizer_options!(
        major_print_level,
        minor_print_level,
        major_iterations_limit,
        minor_iterations_limit,
        major_optimality_tolerance,
        major_feasibility_tolerance,
        minor_feasibility_tolerance,
        derivative_option,
        hessian,
        additional_options
    )
    options = snopt_optimizer_option_pairs(
        major_print_level,
        minor_print_level,
        major_iterations_limit,
        minor_iterations_limit,
        major_optimality_tolerance,
        major_feasibility_tolerance,
        minor_feasibility_tolerance,
        derivative_option,
        hessian,
        additional_options
    )
    validate_snopt_option_pairs_with_library!(options)
    return nothing
end

function heuristic_snopt_workspace_lengths(n::Int, m_eff::Int, neJ::Int)
    n > 0 || throw(ArgumentError("n must be positive, got $n"))
    m_eff > 0 || throw(ArgumentError("m_eff must be positive, got $m_eff"))
    neJ >= 0 || throw(ArgumentError("neJ must be nonnegative, got $neJ"))

    # SNOPT's manual gives 100(m+n) integer and 200(m+n) real slots as only
    # estimates; dense constrained problems need extra room for factorization.
    total = n + m_eff
    leniw = max(
        30500,
        500 + 100 * total,
        500 + 100 * total + 10 * neJ
    )
    lenrw = max(
        m_eff > 1 ? 30500 : 3000,
        500 + 200 * total,
        500 + 500 * total + 20 * neJ
    )
    return leniw, lenrw
end

snopt_workspace_lengths(n::Int, m_eff::Int, neJ::Int) =
    heuristic_snopt_workspace_lengths(n, m_eff, neJ)

function configure_snopt_options!(
        ws,
        opt::SnoptOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing
    )
    for (key, value) in snopt_optimizer_option_pairs(opt; include_print_levels = false)
        apply_snopt_option!(ws, key, value)
    end

    !isnothing(maxiters) && set_option!(ws, "Major iterations limit", Int(maxiters))
    !isnothing(maxtime)  && set_option!(ws, "Time limit", Float64(maxtime))
    !isnothing(abstol)   && set_option!(ws, "Major optimality tolerance", Float64(abstol))
    !isnothing(reltol)   && set_option!(ws, "Minor feasibility tolerance", Float64(reltol))

    set_option!(ws, "Major print level", 0)
    set_option!(ws, "Minor print level", 0)
    set_option!(ws, "Solution = No")

    return ws
end

function snopt_workspace_lengths(
        n::Int,
        m_eff::Int,
        nc::Int,
        J::SparseMatrixCSC,
        opt::SnoptOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing
    )
    neJ = nnz(J)
    mem_ws = initialize("", "", 1000, 1000)
    try
        configure_snopt_options!(mem_ws, opt; maxiters, maxtime, abstol, reltol)
        negCon = nc > 0 ? neJ : 0
        nnCon = nc
        nnJac = nc > 0 ? n : 0
        nnObj = n
        memory = Snopt.snmemb(mem_ws, m_eff, n, neJ, negCon, nnCon, nnJac, nnObj)
        if memory.info == 100 || memory.info == 104
            return memory.miniw, memory.minrw
        end
        error("SNOPT memory estimator failed with info code $(memory.info)")
    finally
        finalize(mem_ws)
    end
end

function map_optimizer_args(
        cache,
        opt::SnoptOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        verbose = false,
        trace_level = SnoptTraceMinimal(),
        store_trace = Val(false),
        progress::Bool = false,
        callback = nothing
    )
    n  = cache.n
    nc = cache.num_cons

    m_eff = nc > 0 ? nc : 1

    user_lb = isnothing(cache.lb) ? fill(-Inf, n) : Vector{Float64}(cache.lb)
    user_ub = isnothing(cache.ub) ? fill(Inf,  n) : Vector{Float64}(cache.ub)
    user_lcons = nc > 0 ? Vector{Float64}(cache.lcons) : Float64[]
    user_ucons = nc > 0 ? Vector{Float64}(cache.ucons) : Float64[]
    lb = snopt_bound_vector(user_lb)
    ub = snopt_bound_vector(user_ub)
    bl = vcat(lb, nc > 0 ? snopt_bound_vector(user_lcons) : [-SNOPT_BOUND_INF])
    bu = vcat(ub, nc > 0 ? snopt_bound_vector(user_ucons) : [SNOPT_BOUND_INF])
    x_ext = zeros(Float64, n + m_eff)
    hs    = zeros(Int32, n + m_eff)

    # Build Jacobian with Int32 indices required by SNOPT.
    # When nc=0, SNOPT requires m>=1; we add a dummy linear constraint row
    # with a single zero coefficient at (1,1) and bounds [-Inf, Inf].
    J = if nc == 0
        SparseMatrixCSC{Float64,Int32}(1, n,
            Int32.(vcat(1, fill(2, n))),
            Int32[1], Float64[0.0])
    elseif cache.J isa SparseMatrixCSC
        SparseMatrixCSC{Float64,Int32}(nc, n,
            Int32.(cache.J.colptr), Int32.(cache.J.rowval),
            zeros(Float64, nnz(cache.J)))
    else
        # Dense Jacobian: all nc×n entries are nonzero
        rowval = repeat(Int32.(1:nc), n)
        colptr = Int32.(range(1; step = nc, length = n + 1))
        SparseMatrixCSC(nc, n, colptr, rowval, zeros(Float64, nc * n))
    end

    leniw, lenrw = snopt_workspace_lengths(
        n, m_eff, nc, J, opt; maxiters, maxtime, abstol, reltol)

    show_output = snopt_show_trace(verbose)
    trace_from_snlog = snopt_log_trace_available() &&
        (show_output || snopt_store_trace(store_trace))
    # Keep SNOPT's own print/summary files disabled. Verbose output is produced
    # synchronously by the Julia callback below, so Windows and single-threaded
    # runs do not buffer the whole SNOPT summary until finalization.
    summfile, reader_task = "", nothing
    ws = initialize("", "", leniw, lenrw)

    logger = SnoptProgressLogger(
        progress, callback, show_output, n, maxiters, cache.iterations;
        trace_level, store_trace, lb = user_lb, ub = user_ub,
        lcon = user_lcons,
        ucon = user_ucons,
        algorithm = opt,
        sense = cache.sense,
        ws_rw = ws.rw,
        trace_from_snlog
    )

    objfun = make_objfun(
        x -> eval_objective(cache, x),
        (g, x) -> eval_objective_gradient(cache, g, x),
        ws.iw;
        callback = logger
    )

    active_confun = if nc > 0
        make_confun(
            (c, x) -> eval_constraint(cache, c, x),
            (jnzval, x) -> eval_constraint_jacobian(cache, jnzval, x),
            J,
            ws.iw;
            callback = logger
        )
    else
        make_dummy_confun()
    end

    configure_snopt_options!(ws, opt; maxiters, maxtime, abstol, reltol)

    return SnoptB(ws, n, nc, m_eff, x_ext, bl, bu, hs, J, 0.0, 0, Float64[], objfun, active_confun), summfile, reader_task, show_output, logger
end

function check_and_convert_maxiters(maxiters::Nothing)
    return nothing
end

function check_and_convert_maxiters(maxiters)
    maxiters isa Integer && !(maxiters isa Bool) ||
        throw(ArgumentError("maxiters must be an integer, got $(repr(maxiters))"))
    maxiters > 0 ||
        throw(ArgumentError("maxiters must be > 0, got $maxiters"))
    return Int(maxiters)
end

function check_and_convert_maxtime(maxtime::Nothing)
    return nothing
end

function check_and_convert_maxtime(maxtime)
    maxtime isa Real && !(maxtime isa Bool) ||
        throw(ArgumentError("maxtime must be a real number, got $(repr(maxtime))"))
    maxtime = Float64(maxtime)
    isfinite(maxtime) ||
        throw(ArgumentError("maxtime must be finite, got $maxtime"))
    maxtime > 0 ||
        throw(ArgumentError("maxtime must be > 0, got $maxtime"))
    return maxtime
end

function map_retcode(inform::Int)
    if inform in (1, 2, 3, 4, 5, 6)
        return SciMLBase.ReturnCode.Success
    elseif inform in (11, 12, 13, 14, 15, 16)
        return SciMLBase.ReturnCode.Infeasible
    elseif inform in (21, 22)
        return SciMLBase.ReturnCode.DivergeFailed
    elseif inform in (31, 32, 33)
        return SciMLBase.ReturnCode.MaxIters
    elseif inform == 34
        return SciMLBase.ReturnCode.MaxTime
    else
        return SciMLBase.ReturnCode.Failure
    end
end

function SciMLBase.__solve(cache::SnoptCache)
    maxiters = check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime  = check_and_convert_maxtime(cache.solver_args.maxtime)

    opt_setup, _, _, _, logger = map_optimizer_args(
        cache,
        cache.opt;
        abstol   = cache.solver_args.abstol,
        reltol   = cache.solver_args.reltol,
        maxiters = maxiters,
        maxtime  = maxtime,
        verbose  = cache.solver_args.show_trace,
        trace_level = cache.solver_args.trace_level,
        store_trace = cache.solver_args.store_trace,
        progress = cache.progress,
        callback = cache.callback
    )

    opt_setup.x[1:cache.n] .= cache.reinit_cache.u0

    start_time = time()
    if logger.trace_from_snlog
        snoptb!(opt_setup; snlog = logger)
    else
        snoptb!(opt_setup)
    end

    # Read results before finalization in case f_snend touches the workspace arrays
    opt_ret    = map_retcode(opt_setup.status)
    minimizer  = opt_setup.ws.x[1:cache.n]
    minimum    = opt_setup.obj_val
    lambda     = copy(opt_setup.lambda[1:cache.n + cache.num_cons])
    iterations = opt_setup.ws.iterations
    major_itns = opt_setup.ws.major_itns
    num_inf    = opt_setup.ws.num_inf
    sum_inf    = opt_setup.ws.sum_inf
    trace_objective = cache.sense === OptimizationBase.MaxSense ? -minimum : minimum
    finish_trace!(logger, opt_setup.status, trace_objective;
        major_iter = major_itns, minor_iter = iterations)
    trace      = stored_trace(logger)

    # Call f_snend immediately rather than relying on the GC finalizer.
    # SNOPT7 has global Fortran state; without this, a second solve in the same
    # process sees stale state and returns instantly with inform=0.
    finalize(opt_setup.ws)

    if cache.progress
        Base.@logmsg(Base.LogLevel(-1), "", progress = 1)
    end

    stats = OptimizationBase.OptimizationStats(;
        time       = time() - start_time,
        iterations = cache.iterations[],
        fevals     = cache.f_calls,
        gevals     = cache.f_grad_calls
    )

    return SciMLBase.build_solution(
        cache,
        cache.opt,
        minimizer,
        minimum;
        original = (inform = opt_setup.status, lambda, iterations, major_itns, num_inf, sum_inf, trace),
        retcode  = opt_ret,
        stats    = stats
    )
end

function SciMLBase.__init(
        prob::OptimizationProblem,
        opt::SnoptOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress::Bool = false,
        show_trace = nothing,
        trace_level = SnoptTraceMinimal(),
        store_trace = Val(false),
        verbose = OptimizationBase.DEFAULT_VERBOSE,
        kwargs...
    )
    # show_trace is SciMLBase's conventional alias for verbose
    final_verbose = isnothing(show_trace) ? verbose : show_trace
    return SnoptCache(prob, opt; maxiters, maxtime, abstol, reltol, progress,
                      verbose = final_verbose, trace_level, store_trace, kwargs...)
end


end # module OptimizationSnopt
