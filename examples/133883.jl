using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using Plots

pars = @parameters a=0.02 b=0.2 c=-50 d=2.0 I=10.0 
vars = @variables u1(t)=-60 u2(t)=-60*b

eqs = [
    D(u1) ~ 0.04 * u1^2 + 5.0 * u1 + 140.0 - u2 + I
    D(u2) ~ a * (b * u1 - u2)
]

root_eqs = [u1 ~ 30]
affect = [
    u1 ~ c
    u2 ~ Pre(u2) + d
]

@mtkcompile sys = System(eqs, t, vars, pars; continuous_events = root_eqs => affect)
prob = ODEProblem(sys, [], (0, 300))



using ModelingToolkitTolerances
sol = solve(prob, Tsit5(); reltol=1e-3,abstol=1e-3)
res = residual(sol)
plot(res)

using ModelingToolkitTolerances
resids = analysis(prob, Tsit5())
plot(resids)

sol = solve(prob, Tsit5(); abstol=1e-3, reltol=1e-3)
plot(sol)

sol = solve(prob, Tsit5(); abstol=1e-9, reltol=1e-12)
plot(sol)


using ModelingToolkitTolerances