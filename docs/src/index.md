# ModelingToolkitTolerances.jl

A Julia package for analyzing tolerance and residual information from ModelingToolkit ODE solutions.

## Overview

ModelingToolkitTolerances.jl provides tools for:

- Computing residuals from ModelingToolkit ODE solutions
- Analyzing tolerance effects on solution accuracy
- Work-precision analysis for solver performance
- CPU timing analysis during integration

## Installation

```julia
using Pkg
Pkg.add("ModelingToolkitTolerances")
```

## Quick Start

```julia
using ModelingToolkitTolerances
using ModelingToolkit, OrdinaryDiffEq

# Create your ModelingToolkit system and solve
# sol = solve(prob, Tsit5())

# Compute residual information
res = residual(sol)

# Run tolerance analysis
results = analysis(prob, Tsit5())
```

## Main Functions

- [`residual`](@ref): Compute residual information from an ODE solution
- [`analysis`](@ref): Run tolerance analysis across multiple abstol/reltol combinations
- [`work_precision`](@ref): Generate work-precision diagrams