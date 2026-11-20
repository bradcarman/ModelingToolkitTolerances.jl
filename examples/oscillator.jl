using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitTolerances
using LinearAlgebra
using Plots

vars = @variables u1(t)=1 u2(t)=1
eqs = [
    D(u1) ~ +500*u2
    D(u2) ~ -500*u1
]
@mtkbuild sys = ODESystem(eqs, t, vars, [])
prob = ODEProblem(sys, [], (0, 1))
sol = solve(prob, ImplicitEuler(); abstol=1e-9, reltol=1e-6)
res_ie = residual(sol)
plot(sol, 10; idxs=u1)

resids = analysis(prob, ImplicitEuler())
plot(resids)

sol = solve(prob, Tsit5())
res_t = residual(sol)
plot(sol, 10; idxs=u1)

res = residual(sol)


plot(res_ie.t, norm(res_ie))
plot!(res_t.t, norm(res_t))