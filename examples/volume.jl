using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
import ModelingToolkitStandardLibrary.Hydraulic.IsothermalCompressible as IC
import ModelingToolkitStandardLibrary.Blocks as B

@component function System(; name)
    pars = []

    systems = @named begin
        fluid = IC.HydraulicFluid()
        src = IC.FixedPressure(; p = 10e5)
        vol = IC.FixedVolume(; vol = 5, p_int=1e5)
        valve = IC.Valve(; Cd = 1e5, minimum_area = 0)
        ramp = B.Ramp(; height = 0.1, duration = 0.1, offset = 0, start_time = 0.1, smooth = true)
    end

    eqs = [connect(fluid, src.port)
           connect(src.port, valve.port_a)
           connect(valve.port_b, vol.port)
           connect(valve.area, ramp.output)]

    ODESystem(eqs, t, [], pars; name, systems)
end

@named sys = System()

using ModelingToolkitDesigner
path = joinpath(@__DIR__, "design") # folder where visualization info is saved and retrieved
design = ODESystemDesign(sys, path);
ModelingToolkitDesigner.view(design)

using CairoMakie
CairoMakie.set_theme!(Theme(;fontsize=12))
fig = ModelingToolkitDesigner.view(design, false)
save(joinpath(path, "sys.png"), fig; resolution=(400,200))