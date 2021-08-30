using F16Model


## TESTING ORIGINAL SYSTEM
# # Define state vector
# # -------------------
# d2r = pi / 180;
# npos = 0;
# epos = 0;
# alt = 10000; # should be in between 5000 ft and 100000 ft
# phi = 0;
# theta = 0;
# psi = 0;
# Vt = 300;
# alp = 0;
# bet = 0;
# p = 0;
# q = 0;
# r = 0;

# x0 = [npos, epos, alt, phi, theta, psi, Vt, alp, bet, p, q, r];

# # Define control vector
# # ---------------------
# T = 9000; # Thrust lbs
# dele = 0; # deg elevator angle
# dail = 0; # deg aileron angle
# drud = 0; # deg rudder angle
# dlef = 0; # deg leading edge flap angle
# u0 = [T, dele, dail, drud, dlef];

# # Trim vehicle at specified altitude and velocity
# h0 = 10000; # ft
# Vt0 = 500;   # ft/s
# xbar, ubar, status, prob = F16Model.Trim(h0, Vt0); # Default is steady-level
# A, B = F16Model.Linearize(xbar, ubar); # Linear system around trim point

## 4-state system for RoA analysis
ind_x = [7, 8, 5, 11]; # V, alpha, theta, q
ind_u = [1, 2]; # Thrust (lbs), elevator angle (deg)
function getLinearModel4x(xbar, ubar)
    # Get 4 state linear model
    A, B = F16Model.Linearize(xbar, ubar) # Linear system around trim point
    A2 = A[ind_x, ind_x]
    B2 = B[ind_x, ind_u]
    return A2, B2
end

# Solve LMI and get nominal controller
using Convex, MosekTools, LinearAlgebra
function getKc(A2, B2)
    # Get nominal stabilizing controller by solving LMI
    t = Convex.Variable()
    Y = Convex.Variable(4, 4)
    W = Convex.Variable(2, 4)
    constraint1 = ((Y - eps() * 1.0I(4)) in :SDP)
    lmiTerm = t * 1.0I(4) - (Y * A2' + A2 * Y + W' * B2' + B2 * W)
    constraint2 = (lmiTerm in :SDP)
    problem = Convex.minimize(t)
    problem.constraints += [constraint1, constraint2]
    Convex.solve!(problem, Mosek.Optimizer)

    Kc = W.value * inv(Y.value)
    return Kc
end
xbar = [
    0.0,
    0.0,
    10000.0,
    0.02290761750153361,
    0.2115123543756087,
    0.0,
    500.0,
    0.21155041298248242,
    0.0007057633870208768,
    -0.015515418839628248,
    0.12976570070063434,
    -0.0031009495374126543,
]
ubar = [
    10072.896496722757,
    -3.93794893001149,
    0.020447222210172724,
    0.019097421123670758,
    12.320513590695686,
]
Kc = [
    -0.20215502439374533 8.92514332974591 2.64510828522257 0.9364780818173069
    -31.83024553378093 1389.558965098923 410.26695969350294 146.98916334744476
]
## TESTING CONTROLLER
## Check if nominal linear system is controlled - Yes.
# using DifferentialEquations
# function f16Dyn(x)
#     u = Kc * x
#     dx = A2 * x + B2 * u
#     return (dx)
# end
# #
# function lSim(x0, tEnd)
#     odeFn(x, p, t) = f16Dyn(x)
#     prob = ODEProblem(odeFn, x0, (0.0, tEnd))
#     sol = solve(prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)
#     return sol
# end
# tEnd = 100.0;
# solCheck = lSim(rand(4), tEnd); # perturb around trim point and check if linear system is controlled
# @show norm(solCheck[1:4, end]) # Looks like linear system is controlled. 


## 4-state nonlinear f16 system around trim point
# function f16Model_4x(x4, xbar, ubar, Kc)
#     # x4 are the states, not perturbations
#     xFull = xbar
#     xFull[ind_x] = x4
#     uFull = ubar
#     uFull[ind_u] += Kc * (x4 - xbar[ind_x])
#     # return xFull, uFull
#     xdotFull = F16Model.Dynamics(xFull, uFull)
#     return xdotFull[ind_x] # return the 4 state dynamics
# end

# function nlSim(x0, tEnd)
#     # function to simulate nonlinear controlled dynamics with initial condition x0 and controller K
#     odeFn2(x, p, t) = f16Model_4x(x, xbar, ubar, Kc)
#     prob = ODEProblem(odeFn2, x0, (0.0, tEnd))
#     sol = solve(prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)
#     return sol
# end
# xbar_4 = xbar[ind_x];
# x0_4 = xbar_4;
# x0_4[1] = 500;
# # f16Model_4x(x0_4,xbar,ubar,Kc)
# # xFull_tmp, uFull_tmp = f16Model_4x(x0_4)
# # F16Model.Dynamics(xFull_tmp, uFull_tmp)
# # F16Model.Dynamics(xbar, ubar)
# sol = nlSim(x0_4, 50);
# @show norm(sol[:, end])
# #
# using PyPlot, LaTeXStrings
# pygui(true);
# function plotTraj(sol, figNum)
#     tSim = sol.t
#     x1Sol = sol[1, :] .- xbar_4[1]
#     x2Sol = sol[2, :] .- xbar_4[2]
#     x3Sol = sol[3, :] .- xbar_4[3]
#     x4Sol = sol[4, :] .- xbar_4[4]
#     # u = [first(phi[2](sol.u[i], optParam)) for i = 1:length(tSim)]

#     figure(figNum)
#     clf()
#     subplot(4, 1, 1)
#     PyPlot.plot(tSim, x1Sol)
#     xlabel("t")
#     ylabel("V")
#     grid("on")
#     subplot(4, 1, 2)
#     PyPlot.plot(tSim, x2Sol)
#     xlabel("t")
#     ylabel(L"$\alpha$")
#     grid("on")
#     subplot(4, 1, 3)
#     PyPlot.plot(tSim, x3Sol)
#     xlabel("t")
#     ylabel(L"$\theta$")
#     grid("on")
#     subplot(4, 1, 4)
#     PyPlot.plot(tSim, x4Sol)
#     xlabel("t")
#     ylabel(L"$\theta$")
#     grid("on")
#     tight_layout()

#     # suptitle("traj_$(titleSuff)");
#     # tight_layout()
# end
# plotTraj(sol, 5)
