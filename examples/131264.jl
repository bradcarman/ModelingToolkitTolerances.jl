using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitStandardLibrary.Blocks: RealOutput, RealInput, Constant
using ModelingToolkitStandardLibrary.Electrical: OnePort, Voltage, Resistance, Resistor, Ground
using ModelingToolkitTolerances
using Plots


@mtkmodel IdealSwitch begin
  @extend v, i = oneport = OnePort()  
  
  @parameters begin 
    Ron = 1e-3, [description = "On resistance magnitud"]
    Roff = 1e6, [description = "Off resistance magnitud"]
  end

  @components begin
    conduction = RealInput() 
  end

  @variables begin
    q(t), [description = "Control Signal "]
    R(t), [description = "Resistance "]
  end
  
  @equations begin
    q ~ conduction.u
    R ~  Ron + q * Roff
    v ~ i * R
  end
end

@component function squareWave(;name, frequency,  amplitude = 1, offset = 0, dutyCycle = 0.5)
    @named output = RealOutput()
    pars = @parameters  amplitude=amplitude frequency=frequency dutyCycle=dutyCycle
    equation =  amplitude * mod1(t, 1/frequency) < dutyCycle/frequency
    eqs = output.u ~ equation
    compose(ODESystem(eqs, t, [], pars; name = name), [output])
end

@mtkmodel R_Test begin
    @parameters begin
        RLoad_param = 10, [description = "Output Load (Ohm)"]
        Vin_param = 200, [description = "Vin Souce (Vdc)"]
        f = 100, [description = "Switch input Frequency (Hz)"]
    end

    @components begin
        switch_input = squareWave(frequency = f, amplitude = 1, offset = 0, dutyCycle = 0.3)
        #switch_input = Sine(frequency = f, amplitude = 0.99, offset = 1.0)
        voltage_input = Constant(k = Vin_param)
        switch = IdealSwitch(Ron = 1e-3, Roff = 1)
        source = Voltage()
        RLoad = Resistor(R = RLoad_param)
        ground = Ground()
    end

    @equations begin 
      connect(voltage_input.output, source.V)
      connect(switch_input.output, switch.conduction)
      connect(source.p, switch.p)
      connect(switch.n,RLoad.p)
      connect(source.n, RLoad.n, ground.g) 
    end
end

@mtkbuild sys = R_Test() additional_passes = [ModelingToolkit.IfLifting]
prob = ODEProblem(sys, [], (0.0, 0.1); guesses = [sys.switch.i => 5.0])
sol = solve(prob; initializealg = ShampineCollocationInit())



prob = ODEProblem(sys, Pair[], (0.0, 0.1); guesses = [sys.switch.i => 5.0], tgrad=false)
sol = solve(prob, Rodas5P(); adaptive=false, dt=1e-4)

prob = ODEProblem(sys, Pair[], (0.0, 0.1); guesses = [sys.switch.i => 5.0], tgrad=true)
sol = solve(prob, Rodas5P(); adaptive=false, dt=1e-4)

prob = ODEProblem(sys, Pair[], (0.0, 0.1); guesses = [sys.switch.i => 5.0], tgrad=false)
sol = solve(prob, Rodas5P(autodiff=false))

prob = ODEProblem(sys, Pair[], (0.0, 0.1); guesses = [sys.switch.i => 5.0], tgrad=false)
sol = solve(prob, Rodas5P(autodiff=true))

prob = ODEProblem(sys, Pair[], (0.0, 0.1); guesses = [sys.switch.i => 5.0], tgrad=true)
sol = solve(prob, Rodas5P(autodiff=false))

prob = ODEProblem(sys, Pair[], (0.0, 0.1); guesses = [sys.switch.i => 5.0], tgrad=true)
sol = solve(prob, Rodas5P(autodiff=true))



pars = @parameters begin
  rho0 = 1000
  beta = 1.2e9
  A = 0.1
  mass = 1000
  e = 1.0
end

vars = @variables begin
  p(t)=100
  m(t), [guess=10]
  x(t)=0.1
  dx(t)=0
end


reg_pow(x, a) = sign(x) * abs(x)^a
rhof(e) = rho0*(1 + reg_pow(p,e)/beta)

eqs = [
  m ~ rhof(e)*x*A
  D(x) ~ dx
  D(dx)*mass ~ p*A
  D(m) ~ t
]

@mtkbuild sys = System(eqs, t, vars, pars)
prob = ODEProblem(sys, [], (0, 1))
sol = solve(prob)
sol = solve(prob, Rodas5P())

prob2 = remake(prob; p = [e=>1.2])
sol2 = solve(prob2, Rodas5P(); adaptive=false, dt=1e-4)

plot(sol; idxs=m)
plot!(sol2; idxs=m)

plot(sol; idxs=p)
plot(sol2; idxs=p)

plot(sol; idxs=dx)
plot!(sol2; idxs=dx)



sol = solve(prob, ImplicitEuler(nlsolve=NLNewton(check_div=false, relax=0.1, always_new=true)); adaptive=false, dt=1e-4)

sol = solve(prob, Rodas5P(); adaptive=false, dt=1e-4)

length(sol)
plot(sol; idxs=m)

using ModelingToolkitTolerances

plot(sol, true; idxs=m)

