using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using Plots

#Parameters

omega_m     = 1.6e6         
N           = 1
Delta_mw    = -0.7           
g0          = 3.6e3#/omega_m 
g_eff       = (g0/sqrt(N))
Gamma       = 60000#/omega_m
Gamma_tilde = 0.0       
Omega_mw    = 0.7           
Q           = 2*(10^5)
gamma_m     = (omega_m/Q) #/ omega_m
N_th        = 6.05
Angle_th    = 0;
tau         = (Gamma/(g_eff^2));
#omega_m     = 1 

function mean_field(du,u,p,t)
    
    
    #p[1]=Delta_mw
    #p[2]=Omega_mw
      
    Sx     =u[1]
    Sy     =u[2]
    Sz     =u[3]
    alpha_x=u[4]
    alpha_y=u[5]
        
    du[1] =  (Delta_mw-4*g_eff*u[4])*u[2]                 - (Gamma+Gamma_tilde)*u[1]
    du[2] = -(Delta_mw-4*g_eff*u[4])*u[1]  -Omega_mw*u[3] - (Gamma+Gamma_tilde)*u[2]
    du[3] =                                 Omega_mw*u[2] - 2*Gamma*(u[3]+N)
    du[4] =  omega_m*u[5]                                 - (gamma_m /2)*u[4]
    du[5] = -omega_m*u[4]                  -g_eff*u[3]    - (gamma_m /2)*u[5]

    
    
    
    #du[1] =  (p[1]-4*g_eff*alpha_x)*Sy          - (Gamma+Gamma_tilde)*Sx
    #du[2] = -(p[1]-4*g_eff*alpha_x)*Sx -p[2]*Sz - (Gamma+Gamma_tilde)*Sy
    #du[3] =                             p[2]*Sy - 2*Gamma*(Sz+N)
    #du[4] =  omega_m*alpha_y                    - (gamma_m /2)*alpha_x
    #du[5] = -omega_m*alpha_x - g_eff*Sz         - (gamma_m /2)*alpha_y
    
end  


u0 = [0;0;-N;sqrt(N_th);0];
time= tau
parameters=[Delta_mw ;Omega_mw ]
tspan = (0.0,time)
prob = ODEProblem(mean_field,u0,tspan,parameters)   #,parameters)

@mtkbuild sys = modelingtoolkitize(prob)
prob = ODEProblem(sys, [], tspan)
sol = solve(prob, Rodas5P())

# ---------------------------------------
# Analysis with ModelingToolkitTolerances
# ---------------------------------------
using ModelingToolkitTolerances
using Plots

# Let's try a low order solver, good for highly stiff problems
resids = analysis(prob, ImplicitEuler(), 0:1e-4:time)
#=
┌────────┬────────────────┬─────────────────┬─────────────────┐
│ abstol │ reltol = 0.001 │ reltol = 1.0e-6 │ reltol = 1.0e-9 │
├────────┼────────────────┼─────────────────┼─────────────────┤
│  0.001 │       0.519476 │          3.5859 │         1.53855 │
│ 1.0e-6 │        284.671 │             NaN │             NaN │
│ 1.0e-9 │        2.27334 │             NaN │             NaN │
└────────┴────────────────┴─────────────────┴─────────────────┘
=#

# Here we see the best tolerance to use is abstol=1e-3, reltol=1e-3
plot(resids; leg=:topleft)
# Note: the max residual result is still quite high (close to 1), therefore the solution is not high quality

# Let's take a look at the result
sol = solve(prob, ImplicitEuler(); abstol=1e-3, reltol=1e-3)
plot(sol; xlims=(0, 5e-6))


# Now let's try a higher order solver, Rodas3 does well here, Rodas5P struggles with this model
resids = analysis(prob, Rodas3(), 0:1e-4:time)
#=
┌────────┬────────────────┬─────────────────┬─────────────────┐
│ abstol │ reltol = 0.001 │ reltol = 1.0e-6 │ reltol = 1.0e-9 │
├────────┼────────────────┼─────────────────┼─────────────────┤
│  0.001 │        2196.16 │         1055.31 │         1086.46 │
│ 1.0e-6 │        610.122 │          2.7878 │         1.00837 │
│ 1.0e-9 │        620.737 │         1.42255 │             NaN │
└────────┴────────────────┴─────────────────┴─────────────────┘
=#

# Here we see the best tolerance to use is abstol=1e-6, reltol=1e-9
plot(resids)
# Note: the max residual is still close to 1, not high quality

sol = solve(prob, Rodas3(); abstol=1e-6, reltol=1e-9)
plot(sol; xlims=(0, 5e-6))



# Rodas4
resids = analysis(prob, Rodas4(), 0:1e-4:time)
#=
┌────────┬────────────────┬─────────────────┬─────────────────┐
│ abstol │ reltol = 0.001 │ reltol = 1.0e-6 │ reltol = 1.0e-9 │
├────────┼────────────────┼─────────────────┼─────────────────┤
│  0.001 │        16310.4 │         3496.79 │         1827.51 │
│ 1.0e-6 │        12888.2 │          65.512 │          28.823 │
│ 1.0e-9 │        16463.4 │         82.4328 │        0.440874 │
└────────┴────────────────┴─────────────────┴─────────────────┘
=#

# Here we see the best tolerance to use is abstol=1e-9, reltol=1e-9
plot(resids)
# Note: the max residual is still close to 1, not high quality

sol = solve(prob, Rodas4(); abstol=1e-9, reltol=1e-9)
plot(sol; xlims=(0, 5e-6))


## Conclusion
# Lowest max residual found is with Rodas4 using abstol & reltol set to 1e-9
