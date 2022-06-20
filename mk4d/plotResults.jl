## Plot the results of the simulation for mk4d
using JLD2, PyPlot, NeuralPDE, DiffEqFlux, ModelingToolkit, LinearAlgebra, Flux, Trapz, Printf, LaTeXStrings, ForwardDiff, Quadrature
pygui(true);
# @variables x1, x2

# Load data 
activFunc = tanh;
dx = 0.05*ones(4,1);
suff = string(activFunc);
nn = 48;

expNum = 5;
cd(@__DIR__);
# fileLoc = "data_quasi/ll_quasi_mk4d_exp$(expNum).jld2";
fileLoc = "data_grid/ll_grid_mk4d_exp$(expNum).jld2";
mkpath("figs_grid/exp$(expNum)")

println("Loading file from ll_grid_exp$(expNum)");
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
title("Loss Functions")
tight_layout();
savefig("figs_grid/exp$(expNum)/loss.png")
## Neural network
dim = 4;
# chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

## PDE and Loss Functions
@parameters x1, x2, x3, x4
@variables  η(..)

xSym = [x1;x2; x3; x4]

# Linear 4D Dynamics
k1 = 1; k2 = 1; k3 = 1;
c1 = 0.4; c2 = 0.4;
linA = Float32.([0 1 0 0;
        -(k1+k2) -c1 k2 0;
        0 0 0 1;
        k2 0 -(k2+k3) -c2]);
f(x) = linA*x;

function g(x::Vector)
    return [0.0f0 0.0f0;1.0f0 0.0f0; 0.0f0 0.0f0; 0.0f0 1.0f0];
end

Q_fpke = 0.4f0; # Q = σ^2
ρ(x) = exp(η(xSym...));

# Equation written directly in η
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G2 = diffC*η(xSym...);

T1_2 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(η(xSym...)))) for i in 1:length(xSym)]); # drift term
# T2_2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G2[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
# T2_2 += sum([(Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j]) - (Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])*η(xSym...) + diffC[i,j]*(Differential(xSym[i])(η(xSym...)))*(Differential(xSym[j])(η(xSym...))) for i in 1:length(xSym), j=1:length(xSym)]); # complete diffusion term
T2_2 = 0.2f0*abs2(Differential(x2)(η(x1, x2, x3, x4))) + 0.2f0*abs2(Differential(x4)(η(x1, x2, x3, x4))) + 0.2f0Differential(x2)(Differential(x2)(η(x1, x2, x3, x4))) + 0.2f0Differential(x4)(Differential(x4)(η(x1, x2, x3, x4)));

Eqn = expand_derivatives(-T1_2+T2_2); 
pdeOrig2 = simplify(Eqn, expand = true) ~ 0.0f0;
pde = pdeOrig2;

initθ = DiffEqFlux.initial_params(chain) #|> gpu;
strategy = NeuralPDE.GridTraining(dx);
derivative = NeuralPDE.get_numeric_derivative();

indvars = xSym
depvars = [η(xSym...)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

_pde_loss_function = NeuralPDE.build_loss_function(pde, indvars, depvars, phi, derivative, integral, chain, initθ, strategy);


## Domains
x1_min = -4.0f0; x1_max = 4.0f0;
x2_min = -4.0f0; x2_max = 4.0f0;
x3_min = -4.0f0; x3_max = 4.0f0;
x4_min = -4.0f0; x4_max = 4.0f0;

## Generate functions to check solution and PDE error
function ρ_pdeErr_fns(optParam)
    function ηNetS(x)
        return first(phi(x, optParam))
    end
    ρNetS(x) = exp(ηNetS(x)) 
    pdeErrFn(x) = first(_pde_loss_function(x, optParam))
    
    return ρNetS, pdeErrFn
end

ρFn, pdeErrFn = ρ_pdeErr_fns(optParam);

@show ρFn(zeros(4))
@show pdeErrFn(zeros(4))



## Compute pdeErr over domain using quadrature (taking forever to compute, investigate alternative?)
# pdeErrQuad(y,p) = pdeErrFn(y);
# prob = QuadratureProblem(pdeErrQuad, [xV_min,xα_min,xθ_min, xq_min], [xV_max, xα_max, xθ_max, xq_max], p = 0);
# sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
# pdeErrTotal = sol.u; @show pdeErrTotal



## Grid discretization (not necessary if using Quadrature)
# nEvalFine = 10; # to be changed
x1Fine = collect(x1_min:dx[1]:x1_max);#range(xV_min, xV_max, length = nEvalFine);
x2Fine = collect(x2_min:dx[2]:x2_max); #range(xα_min, xα_max, length = nEvalFine);
x3Fine = collect(x3_min:dx[3]:x3_max);# range(xθ_min, xθ_max, length = nEvalFine);
x4Fine = collect(x4_min:dx[4]:x4_max); #range(xq_min, xq_max, length = nEvalFine);

# RHOFine = [ρFn([x1d, x2d, x3d, x4d]) for x1d in x1Fine, x2d in x2Fine, x3d in x3Fine, x4d in x4Fine]; # no need to compute true ρ for full state

## Normalize
ρFnQuad(x,p) = ρFn(x); # to obtain normalization constant
prob = QuadratureProblem(ρFnQuad, [x1_min,x2_min,x3_min, x4_min], [x1_max, x2_max, x3_max, x4_max], p = 0);
sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
println("Found normalizing constant: $(sol.u)"); # don't know if it's correct

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
RHO12FineP = RHO12Fine/trapz((x1Fine, x2Fine), RHO12Fine); # normalizing
println("Marginal pdf for x1 and x2 computed.")

## Marginal pdf for the states(V, q)
function ρ14(xV,xq)
    # p = [xV, xα]; y = [xθ, xq]
    ρFnQuad23(y,p) = ρFn([xV;y;xq])
    prob = QuadratureProblem(ρFnQuad23, [xα_min; xθ_min], [xα_max; xθ_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end
# RHO14Fine = [ρ14(xvg, xqg) for xvg in xVFine, xqg in xqFine];
# RHO14Fine = RHO14Fine/sol.u; # normalizing
# println("Marginal pdf for V and q computed.")

## Marginal pdf for the states(α, q)
function ρ24(xα,xq)
    ρFnQuad13(y,p) = ρFn([y[1];xα;y[2];xq])
    prob = QuadratureProblem(ρFnQuad13, [xV_min; xθ_min], [xV_max; xθ_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end
# RHO24Fine = [ρ24(xαg, xqg) for xαg in xαFine, xqg in xqFine];
# RHO24Fine = RHO24Fine/sol.u; # normalizing
# println("Marginal pdf for α and q computed.")

## Marginal pdf for the states(α, θ)
function ρ23(xα, xθ)
    ρFnQuad14(y,p) = ρFn([y[1];xα;xθ;y[2]])
    prob = QuadratureProblem(ρFnQuad14, [xV_min; xq_min], [xV_max; xq_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end
# RHO23Fine = [ρ23(xαg, xθg) for xαg in xαFine, xθg in xθFine];
# RHO23Fine = RHO23Fine/sol.u; # normalizing
# println("Marginal pdf for α and θ computed.")



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
    pcolor(XX, YY, RHOFine, shading = "auto", cmap = "hsv"); colorbar();
    # pcolor(xvFine_XX, xαFine_YY, RHO12Fine, shading = "auto", cmap = "inferno")
    # surf(XX, YY, RHOFine, cmap = "hsv"); #zlabel("Marginal PDF ρ($(label1), $(label2))");
    # colorbar();
    xlabel(label1); ylabel(label2)
    axis("auto")
    title("Marginal PDF ρ($(label1), $(label2))");
    # title("Steady-State Solution (ρ)")

    tight_layout()

end
plotDistErr(12, x1Fine,x2Fine, RHO12FineP, "x1", "x2");
savefig("figs_grid/exp$(expNum)/pdf12.png")
# savefig("figs_quasi/exp$(expNum).png");
# plotDistErr(14, xVFine,xqFine, RHO14Fine, "V (ft/s)", "q (rad/s)");
# plotDistErr(24, xαFine,xqFine, RHO24Fine, "α (rad)", "q (rad/s)");
# plotDistErr(23, xαFine,xθFine, RHO23Fine, "α (rad)", "θ (rad");

# any(isnan.(RHO24Fine))
##
# ρ12_int = trapz((xVFine,xαFine), RHO12Fine); @show ρ12_int
# ρ14_int = trapz((xVFine,xqFine), RHO12Fine); @show ρ14_int
# ρ24_int = trapz((xαFine,xqFine), RHO24Fine); @show ρ24_int
# ρ23_int = trapz((xαFine,xθFine), RHO23Fine); @show ρ23_int

## EXACT SOLUTION
# μ_ssT = zeros(Float32,4);
# Σ_ssT = [0.3333 0 0.16667 0;
#          0 0.5 0 0;
#          0.16667 0 0.3333 0;
#          0 0 0 0.5];

# using Distributions         
# ρ_ssT(xSym) = pdf(MvNormal(μ_ssT, Σ_ssT), xSym);

# # Marginal pdf for the first two states
# using Cubature
# function ρ12T(x1,x2)
#     # p = [xV, xα]; y = [xθ, xq]
#     ρFnQuad34(y,p) = ρ_ssT([x1;x2;y])
#     prob = QuadratureProblem(ρFnQuad34, [x3_min; x4_min], [x3_max; x4_max], p = 0)
#     sol = solve(prob,HCubatureJL(),reltol=1e-6,abstol=1e-6)
#     return sol.u
# end
# RHO12FineT = [ρ12T(x1d, x2d) for x1d in x1Fine, x2d in x2Fine];
# # RHO12FineT2 = RHO12FineT/trapz((x1Fine, x2Fine), RHO12FineT); # normalizing
# println("Marginal pdf for x1 and x2 computed.")
# ##
# plotDistErr(120, x1Fine,x2Fine, RHO12FineT, "x1", "x2");
# title("True Marginal PDF")
# savefig("truePDF.png")
