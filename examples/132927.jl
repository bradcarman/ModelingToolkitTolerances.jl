using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
# using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve
using OrdinaryDiffEq
using NonlinearSolve

@variables begin
    cation1m(t), [guess=1.0]
    anion1m(t), [guess=1.0]
    salt1M(t), [guess=1.0]
    anion2m(t), [guess=1.0]
    salt2M(t), [guess=1.0]
    W(t), [guess=1.0]
    totalcat(t), [guess=2.0]
end
@parameters begin
    maw1=2.0
    maw2=3.0
end
eqs = [
    1 ~ anion1m   / maw1 + anion2m / maw2
    cation1m ~ anion1m  + anion2m
    anion1m * W ~ salt1M
    anion2m * W ~ salt2M
    totalcat ~ salt1M + salt2M
    0 ~ cation1m - anion1m - anion2m
    D(totalcat) ~ 0.0
]
@named aqt = System(eqs, t)

prob = ODEProblem(aqt,
    [
        aqt.totalcat => 2.0,
        # aqt.totalan1 => 1.5,
        # aqt.totalan2 => 2.0,
    ],
    (0.0, 1.0))
sprob = ODEProblem(prob, [], (0, 1))
sol = solve(sprob, NewtonRaphson())


sol = solve(prob, Rodas5P(autodiff=false); adaptive=false, dt=0.1) 


# ┌ Warning: At t=0.0, dt was forced below floating point epsilon 5.0e-324, and step error estimate = NaN.

iprob = prob.f.initialization_data.initializeprob
isys = iprob.f.sys
isol = solve(iprob)

residual = [isol[eqs[i].rhs - eqs[i].lhs] for i=1:8]



sol = solve(prob)
##

sol(0.0; idxs=eqs[2].rhs - eqs[2].lhs)


sol = solve(prob, ImplicitEuler(nlsolve=NLNewton(check_div=false, relax=0.1)); adaptive=false, dt=0.0001)

using Plots
plot(sol)


using ModelingToolkitTolerances

resids = analysis(prob, ImplicitEuler())