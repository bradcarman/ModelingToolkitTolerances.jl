module ModelingToolkitTolerances
using SciMLBase
using ForwardDiff
using ModelingToolkit
using LinearAlgebra
using FilterHelpers
using PrettyTables

export residual, analysis

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

"""
    residual(sol::ODESolution, tms = sol.t; abstol=0.0, reltol=0.0, timing=0.0)

Calculates residual information for `sol::ODESolution` coming from a `ModelingToolkit` model at the times `tms`, which defaults to the solved time points `sol.t`.  Returns as `ResidualInfo` object.

Keyword Arguments (used by `analysis() function`):
- `abstol`: this simply records the `abstol` that was used to create `sol` object.  Since this information is not recorded in an `ODESolution` object it must be specified explicitly for reference.  Used by `analysis()` function.
- `reltol`: same as above, but for `reltol`
- `timing`: this records the corresponding solution time reference.  Used by `analysis()` function.
"""
function residual(sol::ODESolution, tms = sol.t; abstol=0.0, reltol=0.0, timing=0.0)
    prob = sol.prob
    f = prob.f
    #f_oop = f.f_oop

    function f_oop(u, p, t) 
        du = similar(u)
        f(du, u, p, t)
        return du
    end

    p = prob.p
    sys = f.sys
    eqs = full_equations(sys)
    ps = parameters(sys)
    sts = unknowns(sys)
    st_vals = [st => sol(tms; idxs=st).u for st in sts]
    p_vals = [p => sol(0.0; idxs=p) for p in ps]
    n = length(tms)

    differential_vars = Int[]
    algebraic_vars = Int[]
    residuals = Vector{Float64}[]
    rhs_data = hcat([f_oop(sol(tm), p, tm) for tm in tms]...)
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

        residual = rhs_data[i,:] .- lhs_data

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
function analysis(prob::ODEProblem, solver, tms = collect(prob.tspan[1]:1e-3:prob.tspan[2]); kwargs...)
   
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

function no_simplify(sys::ODESystem)

    expanded_sys = expand_connections(sys)
    eqs = equations(expanded_sys)
    vars = unknowns(expanded_sys)
    pars = parameters(expanded_sys)
    iv = ModelingToolkit.independent_variable(expanded_sys)
    defs = ModelingToolkit.defaults(expanded_sys)

    eqs_ = Equation[]
    for eq in eqs
        if typeof(eq.lhs) != ModelingToolkit.Connection && !ModelingToolkit._iszero(eq.lhs) && !ModelingToolkit.isdifferential(eq.lhs)
            push!(eqs_, 0 ~ eq.rhs - eq.lhs)
        else
            push!(eqs_, eq)
        end
    end

    system = ODESystem(eqs_, iv, vars, pars; name=expanded_sys.name, defaults=defs)

    return complete(system)
end


function show_summary(infos::Vector{ResidualInfo})
        header = ["abstol"]
    for reltol in reltols
        push!(header, "reltol = $reltol")
    end
    
    
    
    cols = Vector{Float64}[]
    for reltol in reltols
        col = Float64[]
        for abstol in abstols
            info = filtersingle(x->(x.abstol == abstol) & (x.reltol == reltol), infos)
            if !isnothing(info)
                push!(col, maximum(info.residuals))
            else
                push!(col, NaN)
            end
        end
        push!(cols, col)
    end

    pretty_table(hcat(abstols, cols...); header)
end


Base.show(io::IO, ::MIME"text/plain", infos::Vector{ResidualInfo}) = show_summary(infos)


end # module ModelingToolkitTolerances