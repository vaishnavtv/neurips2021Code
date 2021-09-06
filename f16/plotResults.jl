## Plot the results of the simulation for baseline linear F16
using JLD2,
    PyPlot,
    NeuralPDE,
    ModelingToolkit,
    LinearAlgebra,
    Flux,
    Trapz,
    Printf,
    LaTeXStrings,
    ForwardDiff,
    Quadrature
pygui(true);
# @variables x1, x2

# Load data 
activFunc = tanh;
dx = 0.05;
suff = string(activFunc);
nn = 50;
optFlag = 1;

Q_fpke = 0.3f0; # Q = σ^2

cd(@__DIR__);
fileLoc = "dataQuad/quad_baseline_f16_ADAM_4.jld2";

println("Loading file");
file = jldopen(fileLoc, "r");
optParam = read(file, "optParam");
PDE_losses = read(file, "PDE_losses");
term1_losses = read(file, "term1_losses");
term2_losses = read(file, "term2_losses");
BC_losses = read(file, "BC_losses");
close(file);


## plot losses
figure(1); clf();
nIters = length(PDE_losses)
semilogy(1:nIters, PDE_losses, label= "PDE_losses");
semilogy(1:nIters, term1_losses, label= "term1_losses");
semilogy(1:nIters, term2_losses, label= "term2_losses");
semilogy(1:nIters, BC_losses, label = "BC_losses");
legend();
tight_layout();
# savefig("figs/ADAM_100k.png");
## Neural network
dim = 4;
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

## System Dynamics
include("f16_controller.jl")
function f16Model_4x(x4, xbar, ubar, Kc)
    # nonlinear dynamics of 4-state model with stabilizing controller
    # x4 are the states, not perturbations
    xFull = Vector{Real}(undef, length(xbar))
    xFull .= xbar
    xFull[ind_x] .= Array(x4)#(x4)
    uFull = Vector{Real}(undef, length(ubar))
    uFull .= ubar
    u = (Kc * (Array(x4) .- xbar[ind_x])) # controller
    uFull[ind_u] .+= u
    uFull[ind_u] = contSat(uFull[ind_u])
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

f(x) = f16Model_4x(x, xbar, ubar, Kc) ;

## Generate functions to check solution and PDE error
function ρ_pdeErr_fns(optParam)
    function ηNetS(x)
        return first(phi(x, optParam))
    end
    ρNetS(x) = exp(ηNetS(x)) 
    df(x) = ForwardDiff.jacobian(f, x)
    dη(x) = ForwardDiff.gradient(ηNetS, x)
    d2η(x) = ForwardDiff.jacobian(dη, x)
    
    pdeErrFn(x) = (tr(df(x)) + dot(f(x), dη(x)) - Q_fpke / 2 * (sum(d2η(x)) + sum(dη(x)*dη(x)')))^2
    term1_pdeErrFn(x) = (tr(df(x)) + dot(f(x), dη(x)))^2
    term2_pdeErrFn(x) = (Q_fpke / 2 * (sum(d2η(x)) + sum(dη(x)*dη(x)')))^2
    
    
    # type 2
    fxρ(x) = f(x)*ρNetS(x);
    dfxρ(x) = ForwardDiff.jacobian(fxρ, x);
    d2ρ(x) = Q_fpke/2*ForwardDiff.hessian(ρNetS,x);
    pdeErrFn2(x) = (-tr(dfxρ(x)) + sum(d2ρ(x)))^2;
    return ρNetS, pdeErrFn, term1_pdeErrFn, term2_pdeErrFn
end
# tr(df(xbar[ind_x])) + dot(f(xbar[ind_x]), dη(xbar[ind_x]))
# (sum(d2η(xbar[ind_x])) + sum(dη(xbar[ind_x])*dη(xbar[ind_x])'))
# sum(d2η(xbar[ind_x]))

ρFn, pdeErrFn, term1_pdeErrFn, term2_pdeErrFn = ρ_pdeErr_fns(optParam);
@show ρFn(xbar[ind_x])
@show pdeErrFn(xbar[ind_x])
@show pdeErrFn2(xbar[ind_x])
@show term1_pdeErrFn(xbar[ind_x])
@show term2_pdeErrFn(xbar[ind_x])

## Compute pdeErr over domain using quadrature (taking forever to compute, investigate alternative?)
pdeErrQuad(y,p) = term1_pdeErrFn(y);
prob = QuadratureProblem(pdeErrQuad, [xV_min,xα_min,xθ_min, xq_min], [xV_max, xα_max, xθ_max, xq_max], p = 0);
# sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
# pdeErrTotal = sol.u; @show pdeErrTotal


## Domains
xV_min = 100;
xV_max = 1500;
xα_min = deg2rad(-20);
xα_max = deg2rad(40);
xθ_min = xα_min;
xθ_max = xα_max;
xq_min = -pi/6;
xq_max = pi/6;

# Grid discretization
dV = 100.0; dα = deg2rad(5); 
dθ = dα; dq = deg2rad(5);
dx = 0.1*[dV; dα; dθ; dq]; # grid discretization in V (ft/s), α (rad), θ (rad), q (rad/s)
# nEvalFine = 10; # to be changed
xVFine = collect(xV_min:dx[1]:xV_max);#range(xV_min, xV_max, length = nEvalFine);
xαFine = collect(xα_min:dx[2]:xα_max); #range(xα_min, xα_max, length = nEvalFine);
xθFine = collect(xα_min:dx[3]:xα_max);# range(xθ_min, xθ_max, length = nEvalFine);
xqFine = collect(xq_min:dx[4]:xq_max); #range(xq_min, xq_max, length = nEvalFine);

# RHOFine = [ρFn([xvg, xαg, xθg, xqg]) for xvg in xVFine, xαg in xαFine, xθg in xθFine, xqg in xqFine]; # no need to compute true ρ for full state

## Normalize
ρFnQuad(x,p) = ρFn(x); # to obtain normalization constant
prob = QuadratureProblem(ρFnQuad, [xV_min,xα_min,xθ_min, xq_min], [xV_max, xα_max, xθ_max, xq_max], p = 0);
sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
println("Found normalizing constant.");

## Quadrature technique to obtain marginal pdf, requires integration for every set of states
# Marginal pdf for the first two states(V, α)
function ρ12(xV,xα)
    # p = [xV, xα]; y = [xθ, xq]
    ρFnQuad34(y,p) = ρFn([xV;xα;y])
    prob = QuadratureProblem(ρFnQuad34, [xθ_min; xq_min], [xθ_max; xq_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end
RHO12Fine = [ρ12(xvg, xαg) for xvg in xVFine, xαg in xαFine];
# RHO12Fine = RHO12Fine/sol.u; # normalizing
println("Marginal pdf for V and α computed.")

## Marginal pdf for the states(V, q)
function ρ14(xV,xq)
    # p = [xV, xα]; y = [xθ, xq]
    ρFnQuad23(y,p) = ρFn([xV;y;xq])
    prob = QuadratureProblem(ρFnQuad23, [xα_min; xθ_min], [xα_max; xθ_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end
RHO14Fine = [ρ14(xvg, xqg) for xvg in xVFine, xqg in xqFine];
RHO14Fine = RHO14Fine/sol.u; # normalizing
println("Marginal pdf for V and q computed.")

## Marginal pdf for the states(α, q)
function ρ24(xα,xq)
    ρFnQuad13(y,p) = ρFn([y[1];xα;y[2];xq])
    prob = QuadratureProblem(ρFnQuad13, [xV_min; xθ_min], [xV_max; xθ_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end
RHO24Fine = [ρ24(xαg, xqg) for xαg in xαFine, xqg in xqFine];
RHO24Fine = RHO24Fine/sol.u; # normalizing
println("Marginal pdf for α and q computed.")

## Marginal pdf for the states(α, θ)
function ρ23(xα, xθ)
    ρFnQuad14(y,p) = ρFn([y[1];xα;xθ;y[2]])
    prob = QuadratureProblem(ρFnQuad14, [xV_min; xq_min], [xV_max; xq_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end
RHO23Fine = [ρ23(xαg, xθg) for xαg in xαFine, xθg in xθFine];
RHO23Fine = RHO23Fine/sol.u; # normalizing
println("Marginal pdf for α and θ computed.")



## Plot shown in paper
function plotDistErr(figNum,x1Fine,x2Fine,RHOFine, label1, label2)
    XX = similar(RHOFine);
    YY = similar(RHOFine);

    for i = 1:length(x1Fine), j = 1:length(x2Fine)
        XX[i, j] = x1Fine[i]
        YY[i, j] = x2Fine[j]
    end

    figure(figNum, [8, 4])
    clf()
    pcolor(XX, YY, RHOFine, shading = "auto", cmap = "inferno")
    # pcolor(xvFine_XX, xαFine_YY, RHO12Fine, shading = "auto", cmap = "inferno")
    colorbar()
    xlabel(label1)
    ylabel(label2)
    axis("auto")
    title("Steady-State Solution (ρ)")

    tight_layout()

end
plotDistErr(12, xVFine,xαFine, RHO12Fine, "V (ft/s)", "α (rad)");
plotDistErr(14, xVFine,xqFine, RHO14Fine, "V (ft/s)", "q (rad/s)");
plotDistErr(24, xαFine,xqFine, RHO24Fine, "α (rad)", "q (rad/s)");
plotDistErr(23, xαFine,xθFine, RHO23Fine, "α (rad)", "θ (rad");

# any(isnan.(RHO24Fine))
##
# ρ12_int = trapz((xVFine,xαFine), RHO12Fine); @show ρ12_int
# ρ14_int = trapz((xVFine,xqFine), RHO12Fine); @show ρ14_int
# ρ24_int = trapz((xαFine,xqFine), RHO24Fine); @show ρ24_int
# ρ23_int = trapz((xαFine,xθFine), RHO23Fine); @show ρ23_int