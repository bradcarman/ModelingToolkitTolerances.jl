# Examples

## Basic Residual Analysis

```julia
using ModelingToolkitTolerances
using ModelingToolkit, OrdinaryDiffEq

# Define a simple ODE system
@variables t x(t) y(t)
@parameters a b
D = Differential(t)

eqs = [
    D(x) ~ -a*x + b*y,
    D(y) ~ a*x - b*y
]

@named sys = ODESystem(eqs, t, [x, y], [a, b])
sys = complete(sys)

# Create problem and solve
u0 = [x => 1.0, y => 0.0]
p = [a => 1.0, b => 0.5]
tspan = (0.0, 10.0)
prob = ODEProblem(sys, u0, tspan, p)
sol = solve(prob, Tsit5())

# Analyze residuals
res = residual(sol)
println("Max residual: ", maximum(norm(res)))
```

## Tolerance Analysis

```julia
# Run comprehensive tolerance analysis
results = analysis(prob, Tsit5())

# Display summary table
display(results)
```

## CPU Timing Analysis

```julia
# Track CPU time during solve
callback, tracked_times = get_tracked_time_callback()
sol = solve(prob, Tsit5(); callback=callback)
cpu_timing = process_tracked_time(tracked_times)

# Or use the convenience method
sol, cpu_timing = solve(prob, true, Tsit5())
```