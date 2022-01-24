# moc for f16 with nominal lqr controller
cd(@__DIR__);
include("f16_controller.jl"); # contains trimming inputs, A,B,K for reduced 4-state system

using LinearAlgebra, DifferentialEquations, ForwardDiff, Trapz, Printf, PyPlot
pygui(true);

tEnd = 1.0; tInt = tEnd/10;
propRHO = false; # propagate ρ̇ = - ρ*(∇.f(x)) if true, else propagate η̇ = -tr(∇.f(x)), where η = ln(ρ/ρ0) and η0 = 0.

# Dynamics f(x)
x4bar = xbar[ind_x]; # relevant 4 states 
u1bar = ubar[ind_u]; # relevant control input
function f16Model_4x(x4, xbar, ubar, Kc)
    # nonlinear dynamics of 4-state model with stabilizing controller
    # x4 are the states, not perturbations

    xFull = Vector{Real}(undef, length(xbar))
    xFull .= xbar
    xFull[ind_x] .= (x4)#(x4)

    uFull = Vector{Real}(undef, length(ubar))
    uFull .= ubar
    u = (Kc * (Array(x4) .- xbar[ind_x])) # controller for perturbed system
    uFull[ind_u] .+= u
    uFull[ind_u] = contSat(uFull[ind_u]) # check saturation
    
    xdotFull = F16Model.Dynamics(xFull, uFull)
    
    return xdotFull[ind_x] # return the 4 state dynamics
end

function contSat(u)
    # controller saturation
    if u[2] < -25.0
        u[2] = -25.0
    elseif u[2] > 25.0
        u[2] = 25.0
    end
    return u
end

f(x) = f16Model_4x(x, xbar, ubar, Kc);
df(x) = ForwardDiff.jacobian(f,x);

##
if propRHO
    uDyn(rho,x) = -rho*tr(df(x)); # propagating rho directly
else
    uDyn(rho, x) = -tr(df(x)); # propagating log(rho/rho0)
end

# propagate density using MOC
function nlSimContMOC(x0)
    nx = 4;
    odeFn(xu,p,t) = [f(xu[1:nx]); uDyn(xu[end], xu[1:nx])]
    prob = ODEProblem(odeFn, x0, (0.0, tEnd));
    sol = solve(prob, Tsit5(), saveat = tInt, reltol = 1e-3, abstol = 1e-3);
    return sol.u
end

# create 4D mesh from domain
xV_min = 300; xV_max = 900;
xα_min = deg2rad(-20); xα_max = deg2rad(20);
xθ_min = deg2rad(-20); xθ_max = deg2rad(20);
xq_min = deg2rad(-10); xq_max = deg2rad(10);

nEvalFine = 10; # no. of points along each axis
using LazyGrids
vFine = collect(LinRange(xV_min, xV_max, nEvalFine));
αFine = collect(LinRange(xα_min, xα_max, nEvalFine));
θFine = collect(LinRange(xθ_min, xθ_max, nEvalFine));
qFine = collect(LinRange(xq_min, xq_max, nEvalFine));

(Vg, αg, θg, qg) = ndgrid(vFine,αFine,θFine,qFine);

μ_0 = x4bar;
# Σ_0 = ([100;deg2rad(3)].^2).*1.0I(2); # uncertainty in V,q
Σ_0 = ([10;deg2rad(1);deg2rad(1);deg2rad(1)].^2).*1.0I(4); # initial uncertainty in all states

rho0(x) =  exp(-1 / 2 * (x - μ_0)' * inv(Σ_0) * (x - μ_0)) / (2 * pi * sqrt(det(Σ_0))); # ρ at t0
RHO0 = [rho0([v, a, th, q]) for v in vFine, a in αFine, th in θFine, q in qFine];


if propRHO
    XU_t = [nlSimContMOC([v; a; th; q; rho0([v;a;th;q])]) for v in vFine, a in αFine, th in θFine, q in qFine];
else
    XU_t = [nlSimContMOC([v; a; th; q; 0.0f0]) for v in vFine, a in αFine, th in θFine, q in qFine];
end

tSpan = 0:tInt:tEnd;
for (t, tVal) in enumerate(tSpan)
    local V_grid = [XU_t[i,j,k,l][t][1] for i in 1:nEvalFine, j in 1:nEvalFine, k in 1:nEvalFine, l in 1:nEvalFine];
    local q_grid = [XU_t[i,j,k,l][t][4] for i in 1:nEvalFine, j in 1:nEvalFine, k in 1:nEvalFine, l in 1:nEvalFine];
    if propRHO
        local RHOgrid = [(XU_t[i,j,k,l][t][5]) for i in 1:nEvalFine, j in 1:nEvalFine, k in 1:nEvalFine, l in 1:nEvalFine]; # joint pdf
    else
        local RHOgrid = [RHO0[i,j,k,l]*exp(XU_t[i,j,k,l][t][5]) for i in 1:nEvalFine, j in 1:nEvalFine, k in 1:nEvalFine, l in 1:nEvalFine]; # joint pdf
    end
    # local normC = trapz((V_grid[:,1], q_grid[2,:]), RHOgrid)
    # @show normC;
    # normC_str = @sprintf("|ρ| = %.3f",normC);
    figure(45); clf();
    scatter(V_grid[:,])
    # pcolor(V_grid, q_grid, RHOgrid); colorbar();
    xlabel("x1"); ylabel("x2");
    xlim(xV_min, xV_max);
    ylim(xq_min,  xq_max);
    title(string("t = $(tVal);"));#, normC_str));
    suptitle("Propagating ρ using MOC");
    tight_layout();
    
end

## testing numerical integration
# using Quadrature, Cubature, Cuba,MonteCarloIntegration
# μ_0 = x4bar;
# # Σ_0 = ([100;deg2rad(3)].^2).*1.0I(2); # uncertainty in V,q
# Σ_0 = ([100;deg2rad(10);deg2rad(10);deg2rad(5)].^2).*1.0I(4); # initial uncertainty in all states

# rho0(x) =  exp(-1 / 2 * (x - μ_0)' * inv(Σ_0) * (x - μ_0)) / (2 * pi * sqrt(det(Σ_0))); # ρ at t0
# RHO0 = [rho0([v, a, th, q]) for v in vFine, a in αFine, th in θFine, q in qFine];
# qFn(x,p) = rho0(x);
# prob = QuadratureProblem(qFn, [xV_min, xα_min, xθ_min, xq_min], [xV_max, xα_max, xθ_max, xq_max]);
# sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3); # doesn't work
# # sol = solve(prob, CubatureJLh(), reltol = 1e-3, abstol = 1e-3); # doesn't work
# # sol = solve(prob, CubatureJLp(), reltol = 1e-3, abstol = 1e-3); # doesn't work
# # sol = solve(prob, CubaVegas(), reltol = 1e-3, abstol = 1e-3); # doesn't work
# # sol = solve(prob, CubaSUAVE(), reltol = 1e-3, abstol = 1e-3); # doesn't work
# # sol = solve(prob, CubaDivonne(), reltol = 1e-3, abstol = 1e-3); # doesn't work
# # sol = solve(prob, CubaCuhre(), reltol = 1e-3, abstol = 1e-3); # doesn't work
# # sol = solve(prob, VEGAS(), reltol = 1e-3, abstol = 1e-3); # doesn't work
# @show sol.u


## testing NumericalIntegration
# using NumericalIntegration
# integrate((vFine, αFine, θFine, qFine), RHO0, Trapezoidal())

