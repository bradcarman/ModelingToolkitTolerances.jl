using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitTolerances
using LinearAlgebra
using Plots

vars = @variables x(t)=1
eq = D(x) ~ 0.25*x
@mtkbuild sys = ODESystem([eq], t, vars, [])
prob = ODEProblem(sys, [], (0, 10))
sol = solve(prob, Euler(); dt=2, adaptive=false)
plot(sol; idxs=x)
plot(sol, true; idxs=x)


pars = @parameters T_inf=300; h=0.7; A=1; m=0.1; c_p=1.2
vars = @variables begin
    T(t)=301
end 
eqs = [
    D(T) ~ (h * A) / (m * c_p) * (T_inf - T)
]
@mtkbuild sys = ODESystem(eqs, t, vars, pars)
prob = ODEProblem(sys, [], (0, 10))
sol = solve(prob, Tsit5())

plot(sol; idxs=T)
plot(sol, true; idxs=T)

sol = solve(prob, Tsit5(); reltol=1e-6)

plot(sol; idxs=T)
plot(sol, true; idxs=T)



# Check the residual
times = 0:0.1:10
res = residual(sol)

plot(res.t, abs.(res.residuals))







my_cgrad = cgrad([:white, :red])
resd = zeros(2,10)
resd[:,7] = ones(2)

xs = 0:0.1:1
ys = sin.(xs)

p = plot(xs, ys)
ymin, ymax = Plots.ylims(p)

zs = zeros(2, length(xs))
zs[1,:] = ys .> 0.5
zs[2,:] = ys .> 0.5

heatmap!(xs, [ymin, ymax], zs; cmap=my_cgrad, fillalpha=0.25)
Plots.ylims!(ymin, ymax)




x = 0:0.1:1
plot(x, cos.(x), ylims=(0.5, 1.1))
plot!(twinx(), x, sin.(x),  ylims=(-1.0, 1.0))



y::Tuple = (1.0,2.0,3.0)

function get_index(y::Tuple, value::Float64)
    findfirst(y .== value)
end

@time get_index(y, 3.0)

z::NTuple = (1.0, 2.0, 3.0)

g = NTuple{3, Float64}()