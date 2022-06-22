# Plot steady-state distribution for F18
using JLD2, PyPlot, DiffEqFlux, ModelingToolkit, LinearAlgebra, Flux, Trapz, Printf, LaTeXStrings, ForwardDiff, Quadrature
pygui(true);
cd(@__DIR__);
include("../utils.jl")
include("f18Dyn.jl")
mkpath("figs_ss")
# @variables x1, x2

# Load data 
activFunc = tanh;
nn = 100;

expNum = 7;
fileLoc = "data_ss/exp$(expNum).jld2";

mkpath("figs_ss/exp$(expNum)")


println("Loading file");
file = jldopen(fileLoc, "r");
optParam = read(file, "optParam");
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
close(file);


## plot losses
figure(1); clf();
nIters = length(PDE_losses)
semilogy(1:nIters, PDE_losses, label= "PDE_losses");
semilogy(1:nIters, BC_losses, label = "BC_losses");
legend();
title("Loss Functions Exp $(expNum)")
tight_layout();
savefig("figs_ss/exp$(expNum)/loss.png")


## Neural network
dim = 4;
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1)); # 3 hls
# chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1)); # 2 hls
initθ,re  = Flux.destructure(chain)
phi = (x,θ) -> re(θ)(Array(x))
## Domains
x1_min = -100f0; x1_max = 100f0;
x2_min = deg2rad(-10f0); x2_max = deg2rad(10f0);
x3_min = x2_min ; x3_max = x2_max ;
x4_min = deg2rad(-5f0); x4_max = deg2rad(5f0);

dx = [2.5f0; deg2rad(1f0); deg2rad(1f0); deg2rad(1f0);]; # discretization size used for training

## Generate functions to check solution and PDE error
function ρ_pdeErr_fns(optParam)
    function ηNetS(x)
        return first(phi(x, optParam))
    end
    ρNetS(x) = exp(ηNetS(x)) 
    # pdeErrFn(x) = first(_pde_loss_function(x, optParam))
    
    return ρNetS#, pdeErrFn
end

ρFn = ρ_pdeErr_fns(optParam);

@show ρFn(zeros(4))


## Grid discretization (not necessary if using Quadrature)
# nEvalFine = 10; # to be changed
x1Fine = collect(x1_min:dx[1]:x1_max);#range(x1_min, x1_max, length = nEvalFine);
x2Fine = collect(x2_min:dx[2]:x2_max); #range(x2_min, x2_max, length = nEvalFine);
x3Fine = collect(x3_min:dx[3]:x3_max);# range(x3_min, x3_max, length = nEvalFine);
x4Fine = collect(x4_min:dx[4]:x4_max); #range(x4_min, x4_max, length = nEvalFine);

# RHOFine = [ρFn([x1d, x2d, x3d, x4d]) for x1d in x1Fine, x2d in x2Fine, x3d in x3Fine, x4d in x4Fine]; # no need to compute true ρ for full state

## Normalize
# ρFnQuad(x,p) = ρFn(x); # to obtain normalization constant
# prob = QuadratureProblem(ρFnQuad, [x1_min,x2_min,x3_min, x4_min], [x1_max, x2_max, x3_max, x4_max], p = 0);
# sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
# println("Found normalizing constant: $(sol.u)"); # don't know if it's correct

## Quadrature technique to obtain marginal pdf, requires integration for every set of states
# Marginal pdf for the first two states
function ρ12(x1,x2)
    # p = [xV, xα]; y = [xθ, xq]
    ρFnQuad34(y,p) = ρFn([x1;x2;y])
    prob = QuadratureProblem(ρFnQuad34, [x3_min; x4_min], [x3_max; x4_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end
RHO12Fine = [ρ12(x1d, x2d) for x1d in x1Fine, x2d in x2Fine];
RHO12Fine = RHO12Fine/trapz((x1Fine, x2Fine), RHO12Fine); # normalizing
println("Marginal pdf for V and α computed.")

## Marginal pdf for the states(V, q)
function ρ14(xV,xq)
    # p = [xV, xα]; y = [xθ, xq]
    ρFnQuad23(y,p) = ρFn([xV;y;xq])
    prob = QuadratureProblem(ρFnQuad23, [x2_min; x3_min], [x2_max; x3_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end
RHO14Fine = [ρ14(xvg, xqg) for xvg in x1Fine, xqg in x4Fine];
RHO14Fine = RHO14Fine/sol.u; # normalizing
println("Marginal pdf for V and q computed.")

## Marginal pdf for the states(α, q)
function ρ24(xα,xq)
    ρFnQuad13(y,p) = ρFn([y[1];xα;y[2];xq])
    prob = QuadratureProblem(ρFnQuad13, [x1_min; x3_min], [x1_max; x3_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end
RHO24Fine = [ρ24(xαg, xqg) for xαg in x2Fine, xqg in x4Fine];
RHO24Fine = RHO24Fine/sol.u; # normalizing
println("Marginal pdf for α and q computed.")

## Marginal pdf for the states(α, θ)
function ρ23(xα, xθ)
    ρFnQuad14(y,p) = ρFn([y[1];xα;xθ;y[2]])
    prob = QuadratureProblem(ρFnQuad14, [x1_min; x4_min], [x1_max; x4_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end
RHO23Fine = [ρ23(xαg, xθg) for xαg in x2Fine, xθg in x3Fine];
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

    figure(figNum)#, [8, 4])
    clf()
    pcolor(XX, YY, RHOFine, shading = "auto", cmap = "inferno"); colorbar();
    # pcolor(xvFine_XX, xαFine_YY, RHO12Fine, shading = "auto", cmap = "inferno")
    # surf(XX, YY, RHOFine, cmap = "hsv"); #zlabel("Marginal PDF ρ($(label1), $(label2))");
    xlabel(label1); ylabel(label2)
    axis("auto")
    title("Marginal PDF ρ($(label1), $(label2))");
    # title("Steady-State Solution (ρ)")

    tight_layout()

end
plotDistErr(12, x1Fine,x2Fine, RHO12Fine, "V (ft/s)", "α (rad)");
savefig("figs_ss/exp$(expNum)/x12.png");
plotDistErr(14, x1Fine,x4Fine, RHO14Fine, "V (ft/s)", "q (rad/s)");
savefig("figs_ss/exp$(expNum)/x14.png");
plotDistErr(24, x2Fine,x4Fine, RHO24Fine, "α (rad)", "q (rad/s)");
savefig("figs_ss/exp$(expNum)/x24.png");
plotDistErr(23, x2Fine,x3Fine, RHO23Fine, "α (rad)", "θ (rad)");
savefig("figs_ss/exp$(expNum)/x23.png");

# any(isnan.(RHO24Fine))
##
# ρ12_int = trapz((xVFine,xαFine), RHO12Fine); @show ρ12_int
# ρ14_int = trapz((xVFine,xqFine), RHO12Fine); @show ρ14_int
# ρ24_int = trapz((xαFine,xqFine), RHO24Fine); @show ρ24_int
# ρ23_int = trapz((xαFine,xθFine), RHO23Fine); @show ρ23_int