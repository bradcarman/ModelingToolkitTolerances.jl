module ModelingToolkitTolerances
using SciMLBase
using ForwardDiff
using ModelingToolkit
using LinearAlgebra
using FilterHelpers
using PrettyTables
using DiffEqBase
using DiffEqCallbacks

export residual, scaled_norm, analysis, work_precision, work_precision!, plot_with_residual, plot_with_residual!, solve_with_cpu_time


struct ResidualInfo
    differential_vars::Vector{Int}
    algebraic_vars::Vector{Int}
    residuals::Matrix{Float64}
    rhs::Matrix{Float64}
    lhs::Matrix{Float64}
    t::Vector{Float64}
    abstol::Float64
    reltol::Float64
    timing::Float64
    state_errors::Matrix{Float64}   # implied state error δu (units of each unknown), see `jacobian_state_errors`
    states::Matrix{Float64}         # unknowns u at each time in `t`
    state_error_dt::Float64         # time scale used for the differential rows of `M/dt - J`
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

function default_dt(sol::ODESolution, tms)
    span = length(tms) > 1 ? last(tms) - first(tms) : sol.t[end] - sol.t[1]
    return float(span / max(length(tms) - 1, 1))
end


"""
    residual(sol::ODESolution, tms = default_range(sol); abstol=0.0, reltol=0.0, timing=0.0)

Calculates residual information for `sol::ODESolution` coming from a `ModelingToolkit` model at the times `tms`, which defaults to the solved time points `sol.t`.  Returns as `ResidualInfo` object.

Keyword Arguments (used by `analysis() function`):
- `abstol`: this simply records the `abstol` that was used to create `sol` object.  Since this information is not recorded in an `ODESolution` object it must be specified explicitly for reference.  Used by `analysis()` function.
- `reltol`: same as above, but for `reltol`
- `timing`: this records the corresponding solution time reference.  Used by `analysis()` function.
- `state_error_dt`: time scale `dt` used to map residuals to state errors through `(M/dt - J) δu = r`.  Defaults to the spacing of `tms`.  See `scaled_norm()`.
"""
function residual(sol::ODESolution, tms = default_range(sol); abstol=NaN, reltol=NaN, timing=NaN, state_error_dt = default_dt(sol, tms))
    prob = sol.prob
    f = prob.f
    sys = f.sys
    eqs = equations(sys)
    n = length(tms)
    m = length(eqs)
        
    
    differential_vars = Int[]
    algebraic_vars = Int[]
    ders = Num[]

    
    task_rhs = Threads.@spawn begin
        rhss = [eq.rhs for eq in eqs]
        #TODO: improve the hcat call
        rhss_data = hcat(sol(tms, idxs=rhss).u...)'    
    end
        
    task_lhs = Threads.@spawn begin
        lhss_data = zeros(n,m)
        for (i,eq) in enumerate(eqs)
            lhs = eq.lhs
            
            if ModelingToolkit.isdifferential(lhs)
                push!(differential_vars, i)
                # ModelingToolkit v9 & v10
                # differential_var = lhs.arguments[1]

                # ModelingToolkit v11
                differential_var = lhs.args[1]
                push!(ders, differential_var)
            else
                push!(algebraic_vars, i)
            end
        end

        fs = time -> sol(time; idxs=ders)
        dfs = time -> ForwardDiff.derivative(fs, time)
        for (i,t) in enumerate(tms)
            lhss_data[i,differential_vars] = dfs(t)
        end    
        lhss_data
    end   
    
    rhss_data = fetch(task_rhs)
    lhss_data = fetch(task_lhs)

    residuals = rhss_data .- lhss_data

    errors, states = jacobian_state_errors(sol, tms, residuals; dt = state_error_dt)

    return ResidualInfo(differential_vars, algebraic_vars, residuals, rhss_data, lhss_data, tms, abstol, reltol, timing, errors, states, state_error_dt)
end

"""
    jacobian_state_errors(sol, tms, residuals; dt) -> (errors, states)

Maps the equation residuals (each in its own physical units) back to state space, giving
an implied state error `δu` with the same units as each unknown.

The model is in mass-matrix form `M u' = f(u, p, t)` and the residual is `r = f(u) - M u'`.
One Newton correction of the implicit step `M (u - uₚ)/dt = f(u)` gives

    (M/dt - J) δu = r,      J = ∂f/∂u

The units match: row `i` of `M/dt - J` has units `[rᵢ]/[uⱼ]`, so `δuⱼ` has the units of
`uⱼ`. For algebraic rows (`Mᵢᵢ = 0`) this is the pure Jacobian mapping `δu = J⁻¹ r`.
`M/dt` regularises the differential rows, so the system stays solvable when `J` is
singular. That happens for plain integrators such as `D(x) ~ v` when nothing depends on `x`.
As `dt → 0`, differential rows reduce to `δu ≈ r dt`.

Returns `errors` and `states`, both `length(tms) × length(unknowns)` matrices.
"""
function jacobian_state_errors(sol::ODESolution, tms, residuals::AbstractMatrix; dt)
    prob = sol.prob
    f = prob.f
    p = prob.p
    n = length(prob.u0)
    @assert size(residuals, 2) == n "Jacobian state error requires a square system: $(size(residuals, 2)) equations vs $n unknowns"

    M = f.mass_matrix isa UniformScaling ? Matrix{Float64}(f.mass_matrix, n, n) : Matrix{Float64}(f.mass_matrix)
    Mdt = M ./ dt

    errors = zeros(length(tms), n)
    states = zeros(length(tms), n)

    Threads.@threads for i in eachindex(tms)
        t = tms[i]
        u = Vector{Float64}(sol(t))
        J = if SciMLBase.isinplace(prob)
            ForwardDiff.jacobian((du, x) -> f(du, x, p, t), similar(u), u)
        else
            ForwardDiff.jacobian(x -> f(x, p, t), u)
        end
        W = Mdt .- J
        r = residuals[i, :]
        δu = try
            lu(W) \ r
        catch err
            err isa SingularException || rethrow()
            pinv(W) * r
        end
        errors[i, :] = δu
        states[i, :] = u
    end

    return errors, states
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
    scaled_norm(info::ResidualInfo, vars = all unknowns; abstol = 1e-6, reltol = 1e-3, p = 2) -> Vector{Float64}

A dimensionless summary of the residual at each time in `info.t`.

`LinearAlgebra.norm(info)` adds raw residuals together even when they have different
physical units. `scaled_norm` avoids this. First it maps the residuals to an implied
state error `δu` through the Jacobian (see `jacobian_state_errors`), so each component
has the units of its unknown. Then it weights each component the way the ODE solvers do:

    eᵢ = δuᵢ / (abstolᵢ + reltol * |uᵢ|)

and returns the RMS (`p = 2`, like `DiffEqBase.ODE_DEFAULT_NORM`) or the max (`p = Inf`) over `vars`.

The defaults (`abstol = 1e-6`, `reltol = 1e-3`) match the ODE solver defaults, so a value of
about 1 means the error the residual implies is about as large as the solver's default
error tolerance. Pass `reltol = 1` to get roughly the relative error in the states instead. `abstol` may be a vector with one entry
per unknown. Use this to give each variable its own nominal scale, like Modelica's
`nominal` attribute. All weights must have the units of their unknown, so a scalar
`abstol` is only an approximate floor near `uᵢ ≈ 0`.

`vars` indexes the unknowns.  In a compiled system, equation `i` pairs with unknown `i`, so
`info.differential_vars` and `info.algebraic_vars` can be passed directly.
"""
function scaled_norm(info::ResidualInfo, vars = 1:size(info.state_errors, 2); abstol = 1e-6, reltol = 1e-3, p = 2)
    n = size(info.state_errors, 2)
    atol = abstol isa AbstractVector ? abstol : fill(float(abstol), n)
    @assert length(atol) == n "abstol must be a scalar or have one entry per unknown ($n)"

    norms = Float64[]
    for i in 1:size(info.state_errors, 1)
        e = [info.state_errors[i, j] / (atol[j] + reltol * abs(info.states[i, j])) for j in vars]
        x = isempty(e) ? 0.0 : (p == 2 ? sqrt(sum(abs2, e) / length(e)) : norm(e, p))
        push!(norms, x)
    end
    return norms
end

"""
    analysis(prob::ODEProblem, solver, tms = collect(prob.tspan[1]:1e-3:prob.tspan[2]); kwargs...)

Runs a 3 x 4 study of `abstol` = [1e-3, 1e-6, 1e-9] and `reltol` = [1e-3, 1e-6, 1e-9, 1e-12].  Returns a `Vector{ResidualInfo}` which can be sent to `plot()` and `work_precision()` functions for visual analysis.  A `solver` from `OrdinaryDiffEq` must be passed as the 2nd argument.  The time points used for residual calculation are provided by `tms` and default to the `prob.tspan` spaced by 1ms.  Provided `kwargs...` are passed to the solve


"""
function analysis(prob::ODEProblem, solver, tms = range(prob.tspan[1], prob.tspan[2], 100); kwargs...)
   
    residuals = ResidualInfo[]
    for (i,abstol) in enumerate(abstols)
        for (j,reltol) in enumerate(reltols)
            # solve(prob, solver; abstol, reltol, kwargs...)
            timing = @timed sol = solve(prob, solver; abstol, reltol, kwargs...)
            println(":: [$(round(Int,100*((i-1)*4+j)/12))%] analysis step: abstol=$abstol, reltol=$reltol")
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
function plot_with_residual end
function plot_with_residual! end

function no_simplify(sys::System)
    expanded_sys = expand_connections(sys)
    eqs = equations(expanded_sys)
    vars = unknowns(expanded_sys)
    pars = parameters(expanded_sys)
    ivs = ModelingToolkit.independent_variables(expanded_sys)
    @assert length(ivs) == 1 "Only systems with 1 independent variable are supported"
    iv = ivs[1]
    
    # defaults() is deprecated in v11. It is now split into initial_conditions and bindings.
    ics = ModelingToolkit.initial_conditions(expanded_sys)
    bnds = ModelingToolkit.bindings(expanded_sys)

    eqs_ = Equation[]
    for eq in eqs
        if ModelingToolkit.is_diff_equation(eq)
            new_eq = move_differentials_to_lhs(eq)
            push!(eqs_, new_eq)

        # Removed the `typeof(eq.lhs) != ModelingToolkit.Connection` check. 
        # v11 expands connections to standard equations, and the Connection type is removed.
        elseif !ModelingToolkit._iszero(eq.lhs) && !ModelingToolkit.isdifferential(eq.lhs)
            push!(eqs_, 0 ~ eq.rhs - eq.lhs)
            
        else
            push!(eqs_, eq)
        end
    end

    # Pass the separated initial_conditions and bindings explicitly
    system = System(eqs_, iv, vars, pars; name = ModelingToolkit.get_name(expanded_sys), initial_conditions = ics, bindings = bnds)

    return complete(system)
end

function move_differentials_to_lhs(eq)
    trms = union(terms(eq.rhs), terms(eq.lhs))

    dtrms = SymbolicUtils.BasicSymbolic[]

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
    
    column_labels = ["abstol"]
    for reltol in reltols
        push!(column_labels, "reltol = $reltol")
    end
    
    residuals = map(scaled_norm, infos)
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
                
                val = maximum(scaled_norm(info))

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

    pretty_table(io, hcat(abstols, cols...); column_labels, title="Summary of Max Scaled Residuals", fit_table_in_display_horizontally = false, fit_table_in_display_vertically = false, kwargs...)
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
    solve_with_cpu_time(prob::SciMLBase.AbstractDEProblem, track_cpu_time::Bool, args...; callback = nothing, kwargs...) -> sol, model_time, cpu_time

Returns the `sol::ODESolution` along with corresponding CPU timing information: 
- `model_time::Vector{Float64}`: evaluation times of the model 
- `cpu_time::Vector{Float64}`: corresponding CPU run time at each `model_time` point
"""
function solve_with_cpu_time(prob::SciMLBase.AbstractDEProblem, args...; kwargs...)
    
    time_cb, tracked_times = get_tracked_time_callback()
    
    kwargs_dict = Dict{Symbol, Any}(kwargs)
    if haskey(kwargs_dict, :callback) && !isnothing(kwargs_dict[:callback])
        # Use CallbackSet to combine the user's callback(s) with yours
        user_cb = kwargs_dict[:callback]
        combined_cb = CallbackSet(user_cb, time_cb)
        kwargs_dict[:callback] = combined_cb
    else
        kwargs_dict[:callback] = time_cb
    end
    
    sol = solve(prob, args...; kwargs_dict...)
    
    cpu_timing = process_tracked_time(tracked_times)
    
    return (sol = sol, cpu_timing = cpu_timing)
end

end # module ModelingToolkitTolerances


#=
old slower version...
function residual(sol::ODESolution, tms = default_range(sol); abstol=0.0, reltol=0.0, timing=0.0)
    
    prob = sol.prob
    f = prob.f
    sys = f.sys
    eqs = equations(sys)
    
    differential_vars = Int[]
    algebraic_vars = Int[]
    

    # @info "Collecting RHS data..."
    rhss = [eq.rhs for eq in eqs]
    rhss_data = hcat(sol(tms; idxs=rhss).u...)'
    
    lhss_data = similar(rhss_data)
    ders = Num[]
    # @info "Collecting LHS data..."
    for (i,eq) in enumerate(eqs)

        lhs = eq.lhs
        rhs = eq.rhs

        lhs_data = if ModelingToolkit.isdifferential(lhs)
            push!(differential_vars, i)
            differential_var = lhs.arguments[1]
            fs = time -> sol(time; idxs=differential_var)
            dfs = time -> ForwardDiff.derivative(fs, time)
            dfs.(tms)
        else
            push!(algebraic_vars, i)
            zero.(tms)
        end

        lhss_data[:,i] = lhs_data
    end

    residuals = rhss_data .- lhss_data
    
    return ResidualInfo(differential_vars, algebraic_vars, residuals, rhss_data, lhss_data, tms, abstol, reltol, timing)
end
=#