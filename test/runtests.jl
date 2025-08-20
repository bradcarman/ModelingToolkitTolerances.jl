using ModelingToolkitTolerances
using Test
using ForwardDiff
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Plots

T_inf=300; h=0.7; A=1; m=0.1; c_p=1.2
vars = @variables begin
    T(t)=301
end 
eqs = [
    D(T) ~ (h * A) / (m * c_p) * (T_inf - T)
]
@mtkcompile sys = System(eqs, t, vars, [])
prob = ODEProblem(sys, [], (0, 10))
sol = solve(prob, Tsit5())

Tsol(t) = sol(t; idxs=sys.T)
dTsol(t) = ForwardDiff.derivative(Tsol, t)

resf(t) = ( (h * A) / (m * c_p) * (T_inf - Tsol(t)) ) - ( dTsol(t) )

times = 0:0.1:10
res = residual(sol, times)

@test all([resf(t) == res.residuals[i,1] for (i,t) in enumerate(times)])

resids = analysis(prob, Tsit5());

@test length(resids) == 12
@test resids[1].reltol == 1e-3
@test resids[1].abstol == 1e-3

@test maximum(resids[1].residuals[:,1]) > 1.0
@test maximum(resids[end].residuals[:,1]) < 1e-3


p = plot(res)
@test p isa Plots.Plot
p = plot(resids)
@test p isa Plots.Plot


# Check equation manipulation
@variables x(t)
eq = D(x) ~ x
new_eq = ModelingToolkitTolerances.move_differentials_to_lhs(eq)
@test isequal(new_eq.rhs, x)

eq = 0 ~ D(x) + x
new_eq = ModelingToolkitTolerances.move_differentials_to_lhs(eq)
@test isequal(new_eq.rhs, -x)

eq = 0 ~ 2D(x) + x
new_eq = ModelingToolkitTolerances.move_differentials_to_lhs(eq)
@test isequal(new_eq.rhs, -x/2)


# wall time checker
@time sol, cpu_timing = solve(prob, true, Tsit5());
@test cpu_timing isa ModelingToolkitTolerances.CPUTiming
#=
function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end
u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 50000.0)
prob = ODEProblem(lorenz!, u0, tspan)
sol,cpu_timing = solve(prob, true; console_progress=true);
plot(cpu_timing)
=#

