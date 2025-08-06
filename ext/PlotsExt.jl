module PlotsExt
    using Plots
    using ModelingToolkitTolerances
    using ModelingToolkitTolerances: ResidualInfo, ResidualSettings, SUMMARY
    using FilterHelpers
    using LinearAlgebra

    function Plots.plot(info::ResidualInfo, settings::ResidualSettings = SUMMARY)
        p = Plots.plot(xlabel="time [s]", ylabel="residual")
        
        differential_vars = if settings.differential isa Bool
            if settings.differential
                info.differential_vars
            else
                Int[]
            end
        elseif settings.differential isa Vector
            intersect(info.differential_vars, settings.differential)
        end

        algebraic_vars = if settings.algebraic isa Bool
            if settings.algebraic
                info.algebraic_vars
            else
                Int[]
            end
        elseif settings.algebraic isa Vector
            intersect(info.algebraic_vars, settings.algebraic)
        end

        summary_vars = if settings.summary isa Bool
            if settings.summary
                vcat(info.differential_vars, info.algebraic_vars)
            else
                Int[]
            end
        elseif settings.summary isa Vector
            settings.summary
        end

        for i in differential_vars
            Plots.plot!(p, info.t, LinearAlgebra.norm(info, [i]); linestyle=:dash, label="diff. $i")
        end

        for i in algebraic_vars
            Plots.plot!(p, info.t, LinearAlgebra.norm(info, [i]); label="alg. $i")
        end
        
        if !isempty(summary_vars)
            Plots.plot!(p, info.t, LinearAlgebra.norm(info, summary_vars), width=2, color=:black, label=nothing)
        end

        return p
    end

    # function Plots.plot(infos::Vector{ResidualInfo}, settings::ResidualSettings = SUMMARY)
   
    #     plts = []
    #     for info in infos
    #         p = Plots.plot(info, settings)
    #         Plots.plot!(p, title="abs=$(info.abstol), rel=$(info.reltol)")
    #         push!(plts, p)
    #     end

    #     return Plots.plot(plts...; layout=length(infos), size=(1000,1000))

    # end
    
    const SUPERSCRIPT_DIGITS = collect("⁰¹²³⁴⁵⁶⁷⁸⁹")
    const SUBSCRIPT_DIGITS   = collect("₀₁₂₃₄₅₆₇₈₉")
    superscript(n::Integer) = join(SUPERSCRIPT_DIGITS[begin .+ reverse(digits(n))])
    subscript(n::Integer)   = join(SUBSCRIPT_DIGITS[begin .+ reverse(digits(n))])

    function Plots.plot(infos::Vector{ResidualInfo}; summary=maximum, leg=:bottomright)

        # defaults
        # abstol = 1e-6
        # reltol = 1e-3
   
        abstols = map(x->x.abstol, infos) |> unique
        reltols = map(x->x.reltol, infos) |> unique

        # xtick_labels = "10^" .* string.(round.(Int,log10.(reltols)))
        p= Plots.plot(; ylabel="max residual", xlabel="reltol", xscale=:log10, yscale=:log10, leg, xticks=reltols)
        for abstol in abstols

            resids = filter(x->x.abstol==abstol, infos)


            
            residuals = map(x->summary(LinearAlgebra.norm(x)), resids)
            reltols = map(x->x.reltol, resids)

            exponent = -round(Int,log10(abstol))

            Plots.plot!(p, reltols, residuals; label="abstol = 10⁻" * superscript(exponent), marker=:dot)

        end

        r = filtersingle(x->(x.abstol == 1e-6) & (x.reltol == 1e-3), infos)
        if !isnothing(r)
            Plots.scatter!(p, [1e-3], [summary(r.residuals)]; label="default (abs,rel) = (10⁻⁶, 10⁻³)", marker=:star, markersize=10)
        end

        return p
    end


    # function Plots.plot(infos::Vector{ResidualInfo})

    #     # defaults
    #     # abstol = 1e-6
    #     # reltol = 1e-3
   
    #     abstols = map(x->x.abstol, infos) |> unique
    #     p= Plots.bar(; ylabel="max residual", xlabel="reltol", yscale=:log10, leg=:bottomright, bar_position = :dodge)
    #     for abstol in abstols

    #         resids = filter(x->x.abstol==abstol, infos)

    #         residuals = map(x->maximum(x.residuals), resids)
    #         # reltols = map(x->x.reltol, resids)
    #         reltols = [1,2,3]

    #         Plots.bar!(p, reltols, residuals; label="abstol = " * string(abstol), marker=:dot)

    #     end

    #     # r = filtersingle(x->(x.abstol == 1e-6) & (x.reltol == 1e-3), infos)
    #     # if !isnothing(r)
    #     #     Plots.plot!(p, [1e-3], [maximum(r.residuals)]; label="default tol.", marker=:star, markersize=10)
    #     # end

    #     return p
    # end


    function ModelingToolkitTolerances.work_precision(infos::Vector{ResidualInfo})
        p = Plots.plot()
        return ModelingToolkitTolerances.work_precision!(p, infos)
    end


    function ModelingToolkitTolerances.work_precision!(p::Plots.Plot, infos::Vector{ResidualInfo})
        residuals = map(x->maximum(x.residuals), infos)
        timings = map(x->x.timing, infos)
        
        data = [residuals timings]
        data = filter(x->!isnan(x[1]), eachrow(data))
        data = sort(data; by=first)
        data = hcat(data...)

        return Plots.plot!(p, data[1,:], data[2,:]; marker=:dot, xlabel="max residual", ylabel="time [s]", xscale=:log10, yscale=:log10)
    end


end