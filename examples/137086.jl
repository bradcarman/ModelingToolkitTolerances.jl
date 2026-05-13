using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using Plots
using ModelingToolkitStandardLibrary.Electrical

function ShockleyDiode(; name, I_S = 1e-6, n_ideal = 1.0, T_K = 293.15)
    @parameters begin
        I_s   = I_S
        n_id  = n_ideal
        T     = T_K
        k_B   = 1.380649e-23
        q_e   = 1.602e-19
    end
    @variables begin
        v(t), [guess=0]
        i(t)
    end
    systems = @named begin
        p = Pin()
        n = Pin()
    end
    eqs = [
        v   ~ p.v - n.v,
        i   ~ I_s * (exp(v / (n_id * k_B * T / q_e)) - 1),
        p.i ~  i,
        n.i ~ -i,
    ]
    System(eqs, t; name, systems)
end

function SourceAC(; name, RMSVoltage = 220, Phase = 0, Freq = 50)
    @named p = Pin()
    @named n = Pin()
    @parameters RMSVoltage=RMSVoltage  Phase=Phase  Freq=Freq
    eqs = [
        p.v - n.v ~ (RMSVoltage * √2) * sin(2π * Freq * t + Phase),
        p.i + n.i ~ 0
    ]
    System(eqs, t, [], [RMSVoltage, Phase, Freq]; systems = [p, n], name)
end

function Circuit4(; name,
                    RMSVoltage = 9.0,
                    Freq       = 50.0,
                    R_load     = 10.0,
                    I_S        = 1e-6,
                    n_ideal    = 1.0,
                    T_K        = 293.15)
    @named srcA  = SourceAC(RMSVoltage = RMSVoltage, Freq = Freq, Phase = 0.0)
    @named srcB  = SourceAC(RMSVoltage = RMSVoltage, Freq = Freq, Phase = π)
    @named D1    = ShockleyDiode(I_S = I_S, n_ideal = n_ideal, T_K = T_K)
    @named D2    = ShockleyDiode(I_S = I_S, n_ideal = n_ideal, T_K = T_K)
    @named D3    = ShockleyDiode(I_S = I_S, n_ideal = n_ideal, T_K = T_K)
    @named D4    = ShockleyDiode(I_S = I_S, n_ideal = n_ideal, T_K = T_K)
    @named gnd   = Ground()
    # @named cdc   = Capacitor(C = 1e-6)
    @named rload = Resistor(R = R_load)
    eqs = [
        connect(srcA.p, D1.p, D2.n),
        connect(srcB.p, D3.p, D4.n),
        connect(rload.p, D1.n, D3.n),
        connect(rload.n, D2.p, D4.p),
        connect(srcA.n, srcB.n, gnd.g),
    ]
    System(eqs, t; systems = [srcA, srcB, D1, D2, D3, D4, gnd, rload], name)
end

@named circuit4 = Circuit4(RMSVoltage = 9.0, Freq = 50.0, R_load = 100.0)
sys4  = mtkcompile(circuit4)
prob4 = ODEProblem(sys4, [], (0.0, 0.1))
sol4  = solve(prob4, Rodas5P(); abstol=1e-9, reltol=1e-12) #, dt = 1e-5, adaptive = false)

plot(sol4; idxs=sys4.D3.p.i)


using ModelingToolkitTolerances
resids = analysis(prob4, Rodas5P())