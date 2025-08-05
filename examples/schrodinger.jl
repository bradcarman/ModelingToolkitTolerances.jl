using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using Plots

Ψ = [-0.999998+0.0im, 0.0021739+0.0im]  # eigenstate at t = -100
tlist = collect(range(-100.0, 100; length=1001))

pars = @parameters begin
    Δ = 1.0
	α = 4.6
end

vars = @variables begin
    u1r(t) = real(Ψ[1])
    u2r(t) = real(Ψ[2])
    u1i(t) = imag(Ψ[1])
    u2i(t) = imag(Ψ[2])
end

X = -1im * [
    0.5 * α * t          Δ
    Δ       -0.5 * α * t
]

eqs = [
    D(u1r) ~ u1r*real(X[1,1]) - u1i*imag(X[1,1]) + u2r*real(X[1,2]) - u2i*imag(X[1,2])
    D(u1i) ~ u1i*real(X[1,1]) + u1r*imag(X[1,1]) + u2i*real(X[1,2]) + u2r*imag(X[1,2])
    D(u2r) ~ u1r*real(X[2,1]) - u1i*imag(X[2,1]) + u2r*real(X[2,2]) - u2i*imag(X[2,2])
    D(u2i) ~ u1i*real(X[2,1]) + u1r*imag(X[2,1]) + u2i*real(X[2,2]) + u2r*imag(X[2,2])
]

@mtkbuild sys = ODESystem(eqs, t, vars, pars)
prob = ODEProblem(sys, [], (tlist[1], tlist[end]) )
sol = solve(prob, Tsit5()) #<-- note: saveat not used, as this affects the ForwardDiff.derivative of the ODESolution

# Here we see that default tolerances give a bad result
plot(sol; idxs=[abs2(sys.u1r + 1im*sys.u1i), abs2(sys.u2r + 1im*sys.u2i)])

using ModelingToolkitTolerances

# Here we see the residual is quite significant
res = residual(sol, tlist)
plot(res)

# Here we can see that default tolerance gives the high residual and that setting the reltol to 1e-6 gives an acceptable residual
resids = analysis(prob, Tsit5(), tlist)
plot(resids) 


# solving with reltol set to 1e-6
sol = solve(prob, Tsit5(); reltol=1e-6, saveat=tlist)
plot(sol; idxs=[abs2(sys.u1r + 1im*sys.u1i), abs2(sys.u2r + 1im*sys.u2i)])