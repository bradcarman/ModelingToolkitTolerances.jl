module ModelingToolkitTolerances
using SciMLBase
using ForwardDiff
using ModelingToolkit
using LinearAlgebra
using FilterHelpers
using PrettyTables
using DiffEqBase
using DiffEqCallbacks

export residual, analysis, work_precision

struct ResidualInfo
    differential_vars::Vector{Int}
    algebraic_vars::Vector{Int}
    residuals::Matrix{Float64}
    t::Vector{Float64}
    abstol::Float64
    reltol::Float64
    timing::Float64
end

struct ResidualSettings
    summary::Union{Bool, Vector{Int}}
    differential::Union{Bool, Vector{Int}}
    algebraic::Union{Bool, Vector{Int}}
end
ResidualSettings(eq::Int) = ResidualSettings([eq], false, false)

const SUMMARY = ResidualSettings(true, false, false)
const ALGEBRAIC = ResidualSettings(false, false, true)
const DIFFERENTIAL = ResidualSettings(false, true, false)

const abstols=[1e-3, 1e-6, 1e-9]
const reltols=[1e-3, 1e-6, 1e-9, 1e-12]

function default_range(sol::ODESolution) 
    n = min(Int(1e5), max(1000, 10 * length(sol)))
    return range(sol.t[1], sol.t[end], n)
end

"""
    residual(sol::ODESolution, tms = default_range(sol); abstol=0.0, reltol=0.0, timing=0.0)

Calculates residual information for `sol::ODESolution` coming from a `ModelingToolkit` model at the times `tms`, which defaults to the solved time points `sol.t`.  Returns as `ResidualInfo` object.

Keyword Arguments (used by `analysis() function`):
- `abstol`: this simply records the `abstol` that was used to create `sol` object.  Since this information is not recorded in an `ODESolution` object it must be specified explicitly for reference.  Used by `analysis()` function.
- `reltol`: same as above, but for `reltol`
- `timing`: this records the corresponding solution time reference.  Used by `analysis()` function.
"""
function residual(sol::ODESolution, tms = default_range(sol); abstol=0.0, reltol=0.0, timing=0.0)
    #NOTE: we re-create the ODEProblem so we have the full f function (this seems to be dropped from the ODESolution)
    # prob = ODEProblem(sol.prob.f.sys, sol(0.0), (tms[1], tms[end]); build_initializeprob=false)
    # f = prob.f
    #f_oop = f.f_oop

    # function f_oop(u, p, t) 
    #     du = similar(u)
    #     f(du, u, p, t)
    #     return du
    # end

    prob = sol.prob
    f = prob.f
    p = prob.p
    sys = f.sys
    eqs = full_equations(sys)
    ps = parameters(sys)
    sts = unknowns(sys)
    # st_vals = [st => sol(tms; idxs=st).u for st in sts]
    # p_vals = [p => sol(0.0; idxs=p) for p in ps]
    n = length(tms)

    differential_vars = Int[]
    algebraic_vars = Int[]
    residuals = Vector{Float64}[]
    # rhs_data = hcat([f_oop(sol(tm), p, tm) for tm in tms]...)


    for (i,eq) in enumerate(eqs)

        lhs = eq.lhs
        rhs = eq.rhs
        lhs_data = if ModelingToolkit.isdifferential(lhs)
            push!(differential_vars, i)
            differential_var = lhs.arguments[1]
            df(time) = ForwardDiff.derivative( ξ -> sol(ξ, idxs=differential_var), time)
            df.(tms)
        else
            push!(algebraic_vars, i)
            zero.(tms)
        end
        
        rhs_data = sol(tms; idxs=rhs)

        residual = rhs_data .- lhs_data

        push!(residuals, residual)
    end

    
    return ResidualInfo(differential_vars, algebraic_vars, hcat(residuals...), tms, abstol, reltol, timing)
end

function LinearAlgebra.norm(info::ResidualInfo, vars = vcat(info.algebraic_vars, info.differential_vars))

    norms = Float64[]
    for i=1:size(info.residuals, 1)
        x = norm(info.residuals[i, vars])
        push!(norms, x)
    end

    return norms
end

"""
    analysis(prob::ODEProblem, solver, tms = collect(prob.tspan[1]:1e-3:prob.tspan[2]); kwargs...)

Runs a 3 x 3 study of `abstol` and `reltol` = [1e-3, 1e-6, 1e-9].  Returns a `Vector{ResidualInfo}` which can be sent to `plot()` and `work_precision()` functions for visual analysis.  A `solver` from `OrdinaryDiffEq` must be passed as the 2nd argument.  The time points used for residual calculation are provided by `tms` and default to the `prob.tspan` spaced by 1ms.  Provided `kwargs...` are passed to the solve


"""
function analysis(prob::ODEProblem, solver, tms = range(prob.tspan[1], prob.tspan[2], 100); kwargs...)
   
    residuals = ResidualInfo[]
    for (i,abstol) in enumerate(abstols)
        for (j,reltol) in enumerate(reltols)
            # solve(prob, solver; abstol, reltol, kwargs...)
            timing = @timed sol = solve(prob, solver; abstol, reltol, kwargs...)
            if sol.retcode == ReturnCode.Success
                res = residual(sol, tms; abstol, reltol, timing = timing.time) #TODO: does timing.time include the compile time?
                push!(residuals, res)
            end
        end
    end

    return residuals
end

function work_precision end
function work_precision! end

function no_simplify(sys::System)

    expanded_sys = expand_connections(sys)
    eqs = equations(expanded_sys)
    vars = unknowns(expanded_sys)
    pars = parameters(expanded_sys)
    ivs = ModelingToolkit.independent_variables(expanded_sys)
    @assert length(ivs) == 1 "Only systems with 1 independent variable is supported"
    iv = ivs[1]
    defs = ModelingToolkit.defaults(expanded_sys)

    eqs_ = Equation[]
    for eq in eqs
        
        if ModelingToolkit.is_diff_equation(eq)
            new_eq = move_differentials_to_lhs(eq)
            push!(eqs_, new_eq)

        elseif typeof(eq.lhs) != ModelingToolkit.Connection && !ModelingToolkit._iszero(eq.lhs) && !ModelingToolkit.isdifferential(eq.lhs)
            push!(eqs_, 0 ~ eq.rhs - eq.lhs)
            
        else
            push!(eqs_, eq)
        end
    end

    system = System(eqs_, iv, vars, pars; name=ModelingToolkit.get_name(expanded_sys), defaults=defs)

    return complete(system)
end

function move_differentials_to_lhs(eq)
    trms = union(terms(eq.rhs), terms(eq.lhs))

    dtrms = SymbolicUtils.BasicSymbolic{Real}[]

    for e in [eq.lhs, eq.rhs]
        x = Symbolics.filterchildren(Symbolics.is_derivative, e)
        if !isnothing(x)
            append!(dtrms, x)
        end
    end

    dtrm = unique(dtrms)
    @assert length(dtrm) == 1 "Can only work with systems containing 1 unique differential term per equation: found $eq"
    dtrm = first(dtrm)

    new_rhs = ModelingToolkit.solve_for(eq, dtrm)
    new_lhs = dtrm

    new_eq = new_lhs ~ new_rhs

    return new_eq
end


function show_summary(io, infos::Vector{ResidualInfo}; kwargs...)
    
    header = ["abstol"]
    for reltol in reltols
        push!(header, "reltol = $reltol")
    end
    
    residuals = map(LinearAlgebra.norm, infos)
    max_residuals = map(maximum, residuals)
    best = minimum(max_residuals)

    timings = map(x->x.timing, infos)
    best_timing = minimum(timings)

    cols = Vector{String}[]
    for reltol in reltols
        col = String[]
        for abstol in abstols
            info = filtersingle(x->(x.abstol == abstol) & (x.reltol == reltol), infos)
            if !isnothing(info)
                
                val = maximum(LinearAlgebra.norm(info))

                timing = info.timing

                # sval = formatted(val, :SCI, ndigits=3)
                # stime = formatted(timing, :SCI, ndigits=2)

                sval = string(round(val; sigdigits=3))
                stime = string(round(timing; sigdigits=2))

                if val == best
                    sval = "* $sval *"
                end

                if timing == best_timing
                    stime = "* $stime *"
                end

                push!(col, sval)
            else
                push!(col, "N/A")
            end
        end
        push!(cols, col)
    end

    pretty_table(io, hcat(abstols, cols...); header, title="Summary of Max Residuals", crop=:none, kwargs...)
end


Base.show(io::IO, ::MIME"text/plain", infos::Vector{ResidualInfo}) = show_summary(io, infos)

"""
    get_tracked_time_callback() -> callback, tracked_times

Creates a callback that tracks the CPU time during a solve.  Use `process_tracked_time(tracked_times)` function to extract `model_time` and `cpu_time`

# Example
```julia
using ModelingToolkitTolerances: get_tracked_time_callback, process_tracked_time
callback, tracked_times = get_tracked_time_callback()
sol = solve(prob, args...; callback, kwargs...)
cpu_timing = process_tracked_time(tracked_times)
plot(cpu_timing)
```
"""
function get_tracked_time_callback()
    tracked_times = Vector{Float64}[]

    track_time(u, t, integrator) = begin
        current_time = Base.time_ns()
        push!(tracked_times, [t, current_time])
        nothing
    end 

    callback = FunctionCallingCallback(track_time; func_start=false)  
    return callback, tracked_times
end

struct CPUTiming
    model_time::Vector{Float64}
    cpu_time::Vector{Float64}
end

"""
    process_tracked_time(tracked_times::Vector{Vector{Float64}}) -> model_time, cpu_time

See `get_tracked_time_callback()` for more information
"""
function process_tracked_time(tracked_times::Vector{Vector{Float64}})
    ys = hcat(tracked_times...)
    model_time = ys[1,:]
    cpu_time = [0; diff(ys[2,:])]/1e9
    return CPUTiming(model_time, cpu_time)
end

"""
    solve(prob::SciMLBase.AbstractDEProblem, track_cpu_time::Bool, args...; callback = nothing, kwargs...) -> sol, model_time, cpu_time

Returns the `sol::ODESolution` along with corresponding CPU timing information: 
- `model_time::Vector{Float64}`: evaluation times of the model 
- `cpu_time::Vector{Float64}`: corresponding CPU run time at each `model_time` point
"""
function DiffEqBase.solve(prob::SciMLBase.AbstractDEProblem, track_cpu_time::Bool, args...; callback = nothing, kwargs...)

    !isnothing(callback) && error("Can't use with other callbacks directly.  Use `get_tracked_time_callback()`` to provide a `CallbackSet` with required callbacks")
    
    if track_cpu_time
        callback, tracked_times = get_tracked_time_callback()
        sol = solve(prob, args...; callback, kwargs...)
        cpu_timing = process_tracked_time(tracked_times)
        return sol, cpu_timing
    else
        return solve(prob, args...; kwargs...)
    end
end


end # module ModelingToolkitTolerances