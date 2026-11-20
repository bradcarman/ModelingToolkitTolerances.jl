using ModelingToolkitTolerances
using Test
using ForwardDiff
using LinearAlgebra
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

# Jacobian-scaled (dimensionless) residual
@test size(res.state_errors) == (length(times), 1)
sn = scaled_norm(res)
@test length(sn) == length(times)
@test all(isfinite, sn)
# ODE with mass matrix I: (1/dt - J) δu = r
J = -(h * A) / (m * c_p)
dt = res.state_error_dt
@test res.state_errors[:, 1] ≈ res.residuals[:, 1] ./ (1/dt - J)

# unit invariance: same model with T in mK -> raw norm scales by 1000, scaled_norm does not
@variables T_mK(t)=301e3
@mtkcompile sys_mK = System([D(T_mK) ~ (h * A) / (m * c_p) * (T_inf*1e3 - T_mK)], t, [T_mK], [])
sol_mK = solve(ODEProblem(sys_mK, [], (0, 10)), Tsit5())
res_mK = residual(sol_mK, times)
@test maximum(norm(res_mK)) > 100 * maximum(norm(res))
@test maximum(scaled_norm(res_mK)) ≈ maximum(scaled_norm(res)) rtol=1e-2

# DAE with an algebraic loop: scaled_norm decreases as tolerances tighten
@variables x(t)=0.01 v(t)=0.0 P(t) [guess=0.0] Q(t) [guess=0.0]
@parameters k=1e5 Ap=1e-3 mp=2.0 R=1e8
dae_eqs = [D(x) ~ v, mp*D(v) ~ P*Ap - k*x, Q ~ Ap*v, 0 ~ P - R*Q + 1e3*sin(P*1e-6)]
@mtkcompile dae = System(dae_eqs, t)
dae_prob = ODEProblem(dae, [], (0, 0.1))
dae_times = range(0, 0.1, 201)
# per-unknown abstol: scale floors to the variable magnitudes
atol = [abs(dae_prob.u0[i]) > 0 ? 1e-6 * abs(dae_prob.u0[i]) : 1.0 for i in eachindex(dae_prob.u0)]
coarse = residual(solve(dae_prob, Rodas5P(); abstol=1e-6, reltol=1e-3), dae_times)
fine = residual(solve(dae_prob, Rodas5P(); abstol=1e-12, reltol=1e-9), dae_times)
@test !isempty(coarse.algebraic_vars)
@test maximum(scaled_norm(fine; abstol=atol)) < 1e-2 * maximum(scaled_norm(coarse; abstol=atol))

resids = analysis(prob, Tsit5());

@test length(resids) == 12
@test resids[1].reltol == 1e-3
@test resids[1].abstol == 1e-3

@test maximum(resids[1].residuals[:,1]) > 1.0
@test maximum(resids[end].residuals[:,1]) < 1e-3


p = plot(res; yscale=:identity)
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
@time sol, cpu_timing = solve_with_cpu_time(prob, Tsit5());
@test cpu_timing isa ModelingToolkitTolerances.CPUTiming

p = plot(cpu_timing)
@test p isa Plots.Plot