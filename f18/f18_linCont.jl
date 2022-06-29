# LINEAR STABILIZING CONTROLLER FOR F18 (4-state)
cd(@__DIR__);
include("f18DynNorm.jl")
nx = length(indX);
nu = length(indU);

## get 4-state matrices A and B from matlab around trim point
linA = zeros(Float32,(nx,nx));
linB = zeros(Float32,(nx,nu));

linA[:,1] =    [ -0.0693   -0.0005         0    0.0000]';
linA[:,2] = [-36.7518   -0.2025         0   -0.8667]';
linA[:,3] = [-31.5404    0.0076         0         0]';
linA[:,4] = [0    1.0000    0.8192   -0.1947]';

linA = Float32.(linA);
linB =  Float32.([-7.5602    0.0009
-0.0343   -0.0000
      0         0
-1.7956         0]);

## Solve LMI and get nominal stabilizing controller
using Convex, MosekTools, LinearAlgebra
function getKc_nomStab(A2, B2)
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
Kc_nomStab = getKc_nomStab(linA,linB);

## TESTING CONTROLLER ON LINEAR SYSTEM
# ## Check if nominal linear system is controlled - Yes.
# Kc = Kc_nomStab;
# using DifferentialEquations
# function f18_linDyn(x, Kc, A, B)
#     # linear perturbation dynamics
#     u = Kc * x
#     dx = A * x + B * u
#     return (dx)
# end
# # # #
# function lSim(x0, tEnd, Kc, A, B)
#     odeFn(x, p, t) = f18_linDyn(x, Kc, A, B)
#     prob = ODEProblem(odeFn, x0, (0.0, tEnd))
#     sol = solve(prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)
#     return sol
# end
# tEnd = 5.0;
# # x̃ = [100,pi/6, pi/6, 1].*rand(4); # perturbation around trim point
# vmin = rand()*[-100f0;deg2rad(-10f0);deg2rad(-10f0); deg2rad(-5f0)] ;
# # # @show x̃
# solCheck = lSim(vmin, tEnd, Kc_nomStab, linA, linB); # perturb around trim point and check if linear system is controlled
# @show norm(solCheck[1:4, end]) # Looks like linear system is controlled. 
#
# using Plots
# p1 = plot(solCheck, vars = (0,1))
# p2 = plot(solCheck, vars = (0,2))
# p3 = plot(solCheck, vars = (0,3))
# p4 = plot(solCheck, vars = (0,4))
# plot(p1, p2, p3, p4, layout= (2,2))