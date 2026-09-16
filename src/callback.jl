struct SnoptState
    iter_count::Int
    obj_value::Float64
end

struct SnoptTraceLevel
    trace_mode::Symbol
    print_frequency::Int
    store_frequency::Int
end

const SNOPT_DUAL_INFEASIBILITY_INDEX = 430
const TRACE_HEADER_REPEAT_INTERVAL = 25

function validate_trace_frequency(name::Symbol, value)
    value isa Integer && !(value isa Bool) ||
        throw(ArgumentError("$(name) must be an integer, got $(repr(value))"))
    value > 0 ||
        throw(ArgumentError("$(name) must be positive, got $value"))
    return Int(value)
end

"""
    SnoptTraceMinimal(; print_frequency=1, store_frequency=1)

Record scalar progress values during a solve.

Both frequencies must be positive integers. `print_frequency` controls printed
rows. `store_frequency` controls rows retained in [`SnoptTrace`](@ref).
"""
function SnoptTraceMinimal(; print_frequency = 1, store_frequency = 1)
    print_frequency = validate_trace_frequency(:print_frequency, print_frequency)
    store_frequency = validate_trace_frequency(:store_frequency, store_frequency)
    return SnoptTraceLevel(:minimal, print_frequency, store_frequency)
end

"""
    SnoptTraceAll(; print_frequency=1, store_frequency=1)

Record scalar progress, points, and constraint values during a solve.

Both frequencies must be positive integers. Full traces copy the point and
constraint arrays for each stored row.
"""
function SnoptTraceAll(; print_frequency = 1, store_frequency = 1)
    print_frequency = validate_trace_frequency(:print_frequency, print_frequency)
    store_frequency = validate_trace_frequency(:store_frequency, store_frequency)
    return SnoptTraceLevel(:all, print_frequency, store_frequency)
end

SnoptTraceMinimal(freq::Integer) = SnoptTraceMinimal(;
    print_frequency = freq, store_frequency = freq)
SnoptTraceAll(freq::Integer) = SnoptTraceAll(;
    print_frequency = freq, store_frequency = freq)

"""
    SnoptTraceEntry

One stored SNOPT iteration.

Unavailable numeric values are `NaN`. The final row may use `nothing` for
vectors or status data that SNOPT did not provide.
"""
struct SnoptTraceEntry
    iteration::Int
    major_iter::Int
    minor_iter::Int
    objective::Float64
    step_norm::Float64
    constraint_violation::Float64
    optimality::Float64
    bound_violation::Float64
    x::Union{Nothing, Vector{Float64}}
    c::Union{Nothing, Vector{Float64}}
    status::Union{Nothing, Int}
end

"""
    SnoptTrace

Stored iteration records and the trace level that produced them.

Access a solve trace through `solution.trace` after passing
`store_trace = Val(true)` to `solve`.
"""
struct SnoptTrace
    history::Vector{SnoptTraceEntry}
    trace_level::SnoptTraceLevel
end

function Base.show(io::IO, ::MIME"text/plain", trace::SnoptTrace)
    isempty(trace.history) && return print(io, "Tracing Disabled")
    print_trace_header(io, trace.trace_level)
    for entry in trace.history
        print_trace_entry(io, entry, trace.trace_level)
    end
    return nothing
end

function trace_level_mode(level::SnoptTraceLevel)
    return level.trace_mode
end

function trace_level_mode(level)
    mode = hasproperty(level, :trace_mode) ? getproperty(level, :trace_mode) : Val(:minimal)
    if mode isa Val{:all} || mode === :all
        return :all
    end
    return :minimal
end

function trace_print_frequency(level)
    return hasproperty(level, :print_frequency) ? Int(getproperty(level, :print_frequency)) : 1
end

function trace_store_frequency(level)
    return hasproperty(level, :store_frequency) ? Int(getproperty(level, :store_frequency)) : 1
end

function normalize_trace_level(level)
    mode = trace_level_mode(level)
    print_frequency = trace_print_frequency(level)
    store_frequency = trace_store_frequency(level)
    mode === :all && return SnoptTraceAll(; print_frequency, store_frequency)
    return SnoptTraceMinimal(; print_frequency, store_frequency)
end

snopt_store_trace(store_trace) = false
snopt_store_trace(store_trace::Bool) = store_trace
snopt_store_trace(::Val{true}) = true
snopt_store_trace(::Val{false}) = false

mutable struct SnoptProgressLogger{C, I, A, S}
    progress::Bool
    callback::C
    show_trace::Bool
    store_trace::Bool
    trace_from_snlog::Bool
    trace_io::I
    trace_level::SnoptTraceLevel
    algorithm::A
    sense::S
    n::Int
    maxiters::Union{Nothing, Int}
    iterations::Ref{Int}
    trace_evals::Ref{Int}
    trace_header_printed::Ref{Bool}
    final_trace_printed::Ref{Bool}
    u::Vector{Float64}
    lb::Vector{Float64}
    ub::Vector{Float64}
    lcon::Vector{Float64}
    ucon::Vector{Float64}
    last_x::Vector{Float64}
    has_last_x::Ref{Bool}
    has_last_constraints::Ref{Bool}
    last_constraint_x::Vector{Float64}
    last_constraints::Vector{Float64}
    last_constraint_violation::Ref{Float64}
    last_optimality::Ref{Float64}
    ws_rw::Vector{Float64}
    pending_entry::Ref{Union{Nothing, SnoptTraceEntry}}
    rows_printed::Ref{Int}
    trace_history::Vector{SnoptTraceEntry}
end

function SnoptProgressLogger(
        progress,
        callback,
        trace,
        n,
        maxiters,
        iterations,
        trace_io = stdout;
        trace_level = SnoptTraceMinimal(),
        store_trace = Val(false),
        lb = fill(-Inf, n),
        ub = fill(Inf, n),
        lcon = Float64[],
        ucon = Float64[],
        algorithm = nothing,
        sense = OptimizationBase.MinSense,
        ws_rw = Float64[],
        trace_from_snlog::Bool = false
    )
    return SnoptProgressLogger(
        progress,
        callback,
        trace,
        snopt_store_trace(store_trace),
        trace_from_snlog,
        trace_io,
        normalize_trace_level(trace_level),
        algorithm,
        sense,
        n,
        maxiters,
        iterations,
        Ref(0),
        Ref(false),
        Ref(false),
        zeros(n),
        Float64.(lb),
        Float64.(ub),
        Float64.(lcon),
        Float64.(ucon),
        zeros(n),
        Ref(false),
        Ref(false),
        zeros(n),
        zeros(length(lcon)),
        Ref(NaN),
        Ref(NaN),
        ws_rw isa Vector{Float64} ? ws_rw : Float64.(ws_rw),
        Ref{Union{Nothing, SnoptTraceEntry}}(nothing),
        Ref(0),
        SnoptTraceEntry[]
    )
end

function trace_algorithm_block(algorithm)
    io = IOBuffer()
    println(io, "Algorithm: SnoptOptimizer(")
    @printf io "    %-30s = %s\n" "hessian" repr(algorithm.hessian)
    @printf io "    %-30s = %d\n" "derivative_option" algorithm.derivative_option
    @printf io "    %-30s = %d\n" "major_iterations_limit" algorithm.major_iterations_limit
    @printf io "    %-30s = %.8e\n" "major_optimality_tolerance" algorithm.major_optimality_tolerance
    @printf io "    %-30s = %.8e\n" "major_feasibility_tolerance" algorithm.major_feasibility_tolerance
    println(io, ")")
    return String(take!(io))
end

function print_trace_algorithm(io::IO, algorithm)
    algorithm === nothing && return nothing
    # Base.have_color is `nothing` until Julia resolves the terminal's color
    # support, which never happens for redirected/non-TTY output (CI, Pkg.test).
    # Coalesce to `false` so the IOContext color flag is always a Bool; passing
    # `nothing` makes printstyled throw "non-boolean (Nothing) used in boolean
    # context".
    color = get(io, :color, something(Base.have_color, false))
    printstyled(IOContext(io, :color => color), trace_algorithm_block(algorithm); color = :green)
    println(io)
    return nothing
end

trace_integer(value::Int) = value < 0 ? "-" : string(value)
trace_float(value::Real) = isfinite(value) ? @sprintf("%.8e", Float64(value)) : "-"

function print_trace_header(io::IO, level::SnoptTraceLevel)
    if level.trace_mode === :all
        @printf io "%5s %5s %14s %14s %14s %14s %14s\n" "-----" "-----" "--------------" "--------------" "--------------" "--------------" "--------------"
        @printf io "%5s %5s %14s %14s %14s %14s %14s\n" "Major" "Minor" "Objective" "Constr viol" "Optimality" "Step" "Bound viol"
        @printf io "%5s %5s %14s %14s %14s %14s %14s\n" "-----" "-----" "--------------" "--------------" "--------------" "--------------" "--------------"
    else
        @printf io "%5s %5s %14s %14s %14s %14s\n" "-----" "-----" "--------------" "--------------" "--------------" "--------------"
        @printf io "%5s %5s %14s %14s %14s %14s\n" "Major" "Minor" "Objective" "Constr viol" "Optimality" "Step"
        @printf io "%5s %5s %14s %14s %14s %14s\n" "-----" "-----" "--------------" "--------------" "--------------" "--------------"
    end
    return nothing
end

function print_trace_header(cb::SnoptProgressLogger)
    cb.trace_header_printed[] && return nothing
    print_trace_algorithm(cb.trace_io, cb.algorithm)
    print_trace_header(cb.trace_io, cb.trace_level)
    cb.trace_header_printed[] = true
    return nothing
end

function print_trace_entry(io::IO, entry::SnoptTraceEntry, level::SnoptTraceLevel)
    major = trace_integer(entry.major_iter)
    minor = trace_integer(entry.minor_iter)
    objective = trace_float(entry.objective)
    constraint_violation = trace_float(entry.constraint_violation)
    optimality = trace_float(entry.optimality)
    step_norm = trace_float(entry.step_norm)
    bound_violation = trace_float(entry.bound_violation)

    if entry.iteration < 0
        if level.trace_mode === :all
            @printf io "%5s %5s %14s %14s %14s %14s %14s\n" major minor objective constraint_violation optimality "-" bound_violation
        else
            @printf io "%5s %5s %14s %14s %14s %14s\n" major minor objective constraint_violation optimality "-"
        end
    elseif level.trace_mode === :all
        @printf io "%5s %5s %14s %14s %14s %14s %14s\n" major minor objective constraint_violation optimality step_norm bound_violation
    else
        @printf io "%5s %5s %14s %14s %14s %14s\n" major minor objective constraint_violation optimality step_norm
    end
    return nothing
end

function snopt_workspace_value(values::Vector{Float64}, index::Int)
    return length(values) >= index ? values[index] : NaN
end

function current_optimality(cb::SnoptProgressLogger)
    isfinite(cb.last_optimality[]) && return max(cb.last_optimality[], 0.0)
    value = snopt_workspace_value(cb.ws_rw, SNOPT_DUAL_INFEASIBILITY_INDEX)
    return isfinite(value) ? max(value, 0.0) : NaN
end

function finite_violation(value::Real, lower::Real, upper::Real)
    lower_violation = isfinite(lower) ? max(Float64(lower) - Float64(value), 0.0) : 0.0
    upper_violation = isfinite(upper) ? max(Float64(value) - Float64(upper), 0.0) : 0.0
    return max(lower_violation, upper_violation)
end

function bounds_violation(x::AbstractVector, lb::AbstractVector, ub::AbstractVector)
    violation = 0.0
    for i in eachindex(x)
        violation = max(violation, finite_violation(x[i], lb[i], ub[i]))
    end
    return violation
end

function constraint_violation(c::AbstractVector, lcon::AbstractVector, ucon::AbstractVector)
    violation = 0.0
    for i in eachindex(c)
        violation = max(violation, finite_violation(c[i], lcon[i], ucon[i]))
    end
    return violation
end

function same_trace_point(x::AbstractVector, y::AbstractVector)
    length(x) == length(y) || return false
    scale = 1.0 + max(norm(x, Inf), norm(y, Inf))
    return norm(x .- y, Inf) <= sqrt(eps(Float64)) * scale
end

function trace_show_now(cb::SnoptProgressLogger, entry::SnoptTraceEntry)
    cb.show_trace || return false
    entry.iteration < 0 && return true
    return mod1(entry.iteration, cb.trace_level.print_frequency) == 1
end

function trace_store_now(cb::SnoptProgressLogger, entry::SnoptTraceEntry)
    cb.store_trace || return false
    entry.iteration < 0 && return true
    return mod1(entry.iteration, cb.trace_level.store_frequency) == 1
end

function emit_trace_entry!(cb::SnoptProgressLogger, entry::SnoptTraceEntry)
    if trace_show_now(cb, entry)
        print_trace_header(cb)   # algorithm block + column header on first call
        rows = cb.rows_printed[]
        if entry.iteration >= 0 && rows > 0 && mod(rows, TRACE_HEADER_REPEAT_INTERVAL) == 0
            print_trace_header(cb.trace_io, cb.trace_level)
        end
        print_trace_entry(cb.trace_io, entry, cb.trace_level)
        entry.iteration >= 0 && (cb.rows_printed[] += 1)
        flush(cb.trace_io)
    end
    trace_store_now(cb, entry) && push!(cb.trace_history, stored_trace_entry(cb, entry))
    return nothing
end

function stored_trace_entry(cb::SnoptProgressLogger, entry::SnoptTraceEntry)
    cb.trace_level.trace_mode === :all && return entry
    return SnoptTraceEntry(
        entry.iteration,
        entry.major_iter,
        entry.minor_iter,
        entry.objective,
        entry.step_norm,
        entry.constraint_violation,
        entry.optimality,
        entry.bound_violation,
        nothing,
        nothing,
        entry.status
    )
end

function update_pending_constraint(entry::SnoptTraceEntry, cb::SnoptProgressLogger)
    c_storage = cb.has_last_constraints[] ? copy(cb.last_constraints) : nothing
    x_storage = copy(entry.x::Vector{Float64})
    return SnoptTraceEntry(
        entry.iteration,
        entry.major_iter,
        entry.minor_iter,
        entry.objective,
        entry.step_norm,
        cb.last_constraint_violation[],
        entry.optimality,
        entry.bound_violation,
        x_storage,
        c_storage,
        nothing
    )
end

function flush_pending_trace!(cb::SnoptProgressLogger)
    entry = cb.pending_entry[]
    entry === nothing && return nothing
    emit_trace_entry!(cb, entry)
    cb.pending_entry[] = nothing
    return nothing
end

function trace_iteration_counters(major_itns::Int, minor_itns::Int)
    # Objective/constraint callbacks see SNOPT's workspace before the solver
    # has committed its major/minor counters. Avoid printing those transient
    # zeros as if they were real per-row iteration counts.
    major_itns == 0 && minor_itns == 0 && return -1, -1
    return major_itns, minor_itns
end

function trace_objective!(
        cb::SnoptProgressLogger,
        major_itns::Int,
        minor_itns::Int,
        x::AbstractVector,
        obj::Float64
    )
    (cb.show_trace || cb.store_trace) || return nothing
    if cb.pending_entry[] !== nothing
        flush_pending_trace!(cb)
    end

    step_norm = cb.has_last_x[] ? norm(x .- cb.last_x, 2) : 0.0
    cb.last_x .= x
    cb.has_last_x[] = true
    cb.trace_evals[] += 1

    c_viol = if isempty(cb.lcon)
        0.0
    elseif cb.has_last_constraints[] && same_trace_point(x, cb.last_constraint_x)
        cb.last_constraint_violation[]
    else
        NaN
    end
    b_viol = bounds_violation(x, cb.lb, cb.ub)
    x_storage = copy(x)
    c_storage = cb.has_last_constraints[] &&
        same_trace_point(x, cb.last_constraint_x) ? copy(cb.last_constraints) : nothing
    opt = current_optimality(cb)
    trace_major_itns, trace_minor_itns = trace_iteration_counters(major_itns, minor_itns)

    entry = SnoptTraceEntry(
        cb.trace_evals[],
        trace_major_itns,
        trace_minor_itns,
        obj,
        step_norm,
        c_viol,
        opt,
        b_viol,
        x_storage,
        c_storage,
        nothing
    )

    if isempty(cb.lcon) || (!isnan(c_viol))
        emit_trace_entry!(cb, entry)
    else
        cb.pending_entry[] = entry
    end
    return nothing
end

function trace_snlog!(cb::SnoptProgressLogger, event)
    (cb.show_trace || cb.store_trace) || return nothing
    cb.trace_from_snlog || return nothing
    flush_pending_trace!(cb)

    nx = min(cb.n, length(event.x))
    x = event.x[1:nx]
    nx == cb.n && copyto!(cb.u, x)
    if nx == cb.n
        copyto!(cb.last_x, x)
        cb.has_last_x[] = true
    end

    c_storage = length(event.fcon) >= length(cb.lcon) ?
        copy(event.fcon[1:length(cb.lcon)]) : nothing
    c_viol = event.primal_infeasibility
    cb.last_constraint_violation[] = c_viol
    if c_storage !== nothing
        copyto!(cb.last_constraints, c_storage)
        cb.has_last_constraints[] = true
    end
    cb.last_optimality[] = event.dual_infeasibility
    cb.iterations[] = event.major_iter
    cb.trace_evals[] += 1

    objective = cb.sense === OptimizationBase.MaxSense ? -event.objective : event.objective
    entry = SnoptTraceEntry(
        cb.trace_evals[],
        event.major_iter,
        event.minor_iter,
        objective,
        event.step,
        c_viol,
        event.dual_infeasibility,
        nx == cb.n ? bounds_violation(x, cb.lb, cb.ub) : event.max_violation,
        nx == cb.n ? copy(x) : nothing,
        c_storage,
        nothing
    )
    emit_trace_entry!(cb, entry)
    return nothing
end

function (cb::SnoptProgressLogger)(event::SNOPT.SnoptMajorLog)
    trace_snlog!(cb, event)
    return true
end

function trace_constraints!(
        cb::SnoptProgressLogger,
        x::AbstractVector,
        c::AbstractVector
    )
    isempty(cb.lcon) && return nothing
    copyto!(cb.last_constraint_x, x[1:cb.n])
    copyto!(cb.last_constraints, c)
    cb.last_constraint_violation[] = constraint_violation(c, cb.lcon, cb.ucon)
    cb.has_last_constraints[] = true

    entry = cb.pending_entry[]
    if entry !== nothing && entry.x !== nothing &&
            same_trace_point(entry.x, cb.last_constraint_x)
        emit_trace_entry!(cb, update_pending_constraint(entry, cb))
        cb.pending_entry[] = nothing
    end
    return nothing
end

function finish_trace!(
        cb::SnoptProgressLogger,
        status::Int,
        objective::Float64;
        major_iter::Int = cb.iterations[],
        minor_iter::Int = 0
    )
    (cb.show_trace || cb.store_trace) || return nothing
    cb.final_trace_printed[] && return nothing
    flush_pending_trace!(cb)
    final_entry = SnoptTraceEntry(
        -1,
        major_iter,
        minor_iter,
        objective,
        NaN,
        (cb.trace_from_snlog || cb.has_last_constraints[]) ? cb.last_constraint_violation[] : 0.0,
        current_optimality(cb),
        cb.has_last_x[] ? bounds_violation(cb.last_x, cb.lb, cb.ub) : NaN,
        nothing,
        nothing,
        status
    )
    emit_trace_entry!(cb, final_entry)
    cb.final_trace_printed[] = true
    return nothing
end

function stored_trace(cb::SnoptProgressLogger)
    cb.store_trace || return nothing
    return SnoptTrace(copy(cb.trace_history), cb.trace_level)
end

function (cb::SnoptProgressLogger)(event::NamedTuple)
    if event.kind === :constraint
        trace_constraints!(cb, event.x, event.c)
        return true
    elseif event.kind !== :objective
        return true
    end

    major_itns = event.major_iter
    minor_itns = event.minor_iter
    cb.iterations[] = major_itns
    copyto!(cb.u, event.x[1:cb.n])
    raw_obj = event.f
    sense = haskey(event, :sense) ? event.sense : cb.sense
    obj = sense === OptimizationBase.MaxSense ? -raw_obj : raw_obj

    if cb.progress && !isnothing(cb.maxiters)
        Base.@logmsg(Base.LogLevel(-1), "",
            progress = major_itns / cb.maxiters)
    end

    cb.trace_from_snlog || trace_objective!(cb, major_itns, minor_itns, cb.u, obj)

    if !isnothing(cb.callback)
        original = SnoptState(major_itns, raw_obj)
        opt_state = OptimizationBase.OptimizationState(;
            iter = major_itns, u = cb.u, objective = obj, original
        )
        return !cb.callback(opt_state, obj)  # user returns true to stop -> we return false
    end
    return true
end

function (cb::SnoptProgressLogger)(major_itns::Int, x::AbstractVector, f::Float64, sense)
    event = (kind = :objective, major_iter = major_itns, minor_iter = 0,
             x = x, f = f, sense = sense)
    return cb(event)
end
