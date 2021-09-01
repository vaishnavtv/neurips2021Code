using F16Model

# # Trim vehicle at specified altitude and velocity
h0 = 10000; # ft
Vt0 = 500;   # ft/s
# xbar, ubar, _, _ = F16Model.Trim(h0, Vt0); # Default is steady-level

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
# A2, B2 = getLinearModel4x(xbar, ubar)

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
# Kc_nomStab = getKc(A2,B2);

## Solve LMI and get nominal lqr controller
Q = Diagonal([1e-1; 1e2; 1e0; 1e-2]); R = Diagonal([1e-3; 1e2]);
function getKc_lqr(A2,B2)
    # Get lqr controller
    Y = Convex.Variable(4,4)
    lmiTerm = -[Y*A2' + A2*Y - B2*inv(R)*B2' Y*sqrt(Q);
                sqrt(Q)*Y   -1.0I(4) ];
    constraint = (lmiTerm in :SDP)
    problem = Convex.maximize(tr(Y));
    problem.constraints += constraint;
    Convex.solve!(problem, Mosek.Optimizer)
    
    Kc = -inv(R)*B2'*inv(Y.value);
    return Kc
end
# Kc_lqr = getKc_lqr(A2,B2);
# Kc = Kc_lqr;

#region
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
];
ubar = [
    10072.896496722757,
    -3.93794893001149,
    0.020447222210172724,
    0.019097421123670758,
    12.320513590695686,
];
Kc = [-0.5470758598111812 103.66925411758012 31.024026394129915 47.40198717690881; -0.03251963249654607 32.11378665734905 3.7426770697698895 14.969332380810693];
#endregion


## TESTING CONTROLLER ON LINEAR SYSTEM
## Check if nominal linear system is controlled - Yes.
# Kc = Kc_lqr;
using DifferentialEquations
function f16_linDyn(x)
    # linear perturbation dynamics
    u = Kc * x
    dx = A2 * x + B2 * u
    return (dx)
end
# #
function lSim(x0, tEnd)
    odeFn(x, p, t) = f16_linDyn(x)
    prob = ODEProblem(odeFn, x0, (0.0, tEnd))
    sol = solve(prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)
    return sol
end
# tEnd = 100.0;
# x̃ = [100,pi/6, pi/6, 1].*rand(4); # perturbation around trim point
# @show x̃
# solCheck = lSim(x̃, tEnd); # perturb around trim point and check if linear system is controlled
# @show norm(solCheck[1:4, end]) # Looks like linear system is controlled. 
