## Plot the results of the simulation for mk4d
using JLD2, PyPlot, NeuralPDE, DiffEqFlux, ModelingToolkit, LinearAlgebra, Flux, Trapz, Printf, LaTeXStrings, ForwardDiff, Quadrature
pygui(true);
# @variables x1, x2

# Load data 
activFunc = tanh;
dx = 0.25f0*ones(4,1);
suff = string(activFunc);
nn = 48;

expNum = 1;
cd(@__DIR__);
fileLoc = "data_grid/ll_grid_zhang4d_exp$(expNum).jld2";
mkpath("figs_grid/exp$(expNum)")

println("Loading file from ll_grid_zhang4d_exp$(expNum)");
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
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
# chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

## PDE and Loss Functions
@parameters x1, x2, x3, x4
@variables  η(..)

xSym = [x1;x2; x3; x4]

## 4D Dynamics
a = 0.5f0; b = 1f0; k1 = -0.5f0; k2 = k1;
ϵ = 0.5f0; λ1 = 0.25f0; λ2 = 0.125f0; μ = 0.375f0;
M = 1f0; varI = 1f0;

vFn(x1,x2) = k1*x1^2 + k2*x2^2 + ϵ*(λ1*x1^4 + λ2*x2^4 + μ*x1^2*x2^2)
dvdx1_expr = Symbolics.derivative(vFn(x1,x2), x1);
dvdx1_fn(y1,y2) = substitute(dvdx1_expr, Dict([x1=>y1, x2=>y2]));
dvdx2_expr = Symbolics.derivative(vFn(x1,x2), x2);
dvdx2_fn(y1,y2) = substitute(dvdx2_expr, Dict([x1=>y1, x2=>y2]));

function f(x)
    output = [x[3]; x[4];
             -a*x[3] - 1/M*dvdx1_fn(x[1],x[2])
             -b*x[4] - 1/varI*dvdx2_fn(x[1],x[2])]
    return output
end

function g(x::Vector)
    return [0.0f0 0.0f0;0.0f0 0.0f0; 1.0f0 0.0f0; 0.0f0 1.0f0];
end


Q_fpke = [2f0 0f0; 0f0 4f0;]; # Q = σ^2
ρ(x) = exp(η(xSym...));

# Equation written directly in η
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G2 = diffC*η(xSym...);

T1_2 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(η(xSym...)))) for i in 1:length(xSym)]); # drift term
T2_2 = 2.0f0*abs2(Differential(x4)(η(x1, x2, x3, x4))) + 2.0f0*Differential(x4)(Differential(x4)(η(x1, x2, x3, x4))) + abs2(Differential(x3)(η(x1, x2, x3, x4))) + Differential(x3)(Differential(x3)(η(x1, x2, x3, x4)))

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
x1_min = -5.0f0; x1_max = 5.0f0;
x2_min = -5.0f0; x2_max = 5.0f0;
x3_min = -5.0f0; x3_max = 5.0f0;
x4_min = -5.0f0; x4_max = 5.0f0;

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

## Grid discretization (not necessary if using Quadrature)
# nEvalFine = 10; # to be changed
x1Fine = collect(x1_min:dx[1]:x1_max);#range(xV_min, xV_max, length = nEvalFine);
x2Fine = collect(x2_min:dx[2]:x2_max); #range(xα_min, xα_max, length = nEvalFine);
x3Fine = collect(x3_min:dx[3]:x3_max);# range(xθ_min, xθ_max, length = nEvalFine);
x4Fine = collect(x4_min:dx[4]:x4_max); #range(xq_min, xq_max, length = nEvalFine);


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
RHO12FineP = RHO12Fine/trapz((x1Fine, x2Fine), RHO12Fine); # normalizing
println("Marginal pdf for x1 and x2 computed.")

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
    pcolor(XX, YY, RHOFine, shading = "auto"); colorbar();
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

## EXACT SOLUTION
varT = Q_fpke[1]/2*M/a
ρ_ssT(xSym) = exp(-1/(2*varT)*vFn(xSym[1],xSym[2]) - a/Q_fpke[1]*xSym[3]^2 - b/Q_fpke[end]*xSym[4]^2)

# Marginal pdf for the first two states
using Cubature
function ρ12T(x1,x2)
    # p = [xV, xα]; y = [xθ, xq]
    ρFnQuad34(y,p) = ρ_ssT([x1;x2;y])
    prob = QuadratureProblem(ρFnQuad34, [x3_min; x4_min], [x3_max; x4_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-6,abstol=1e-6)
    return sol.u
end
RHO12FineT = [ρ12T(x1d, x2d) for x1d in x1Fine, x2d in x2Fine];
RHO12FineT2 = RHO12FineT/trapz((x1Fine, x2Fine), RHO12FineT); # normalizing
println("True Marginal pdf for x1 and x2 computed.")
##
plotDistErr(120, x1Fine,x2Fine, RHO12FineT2, "x1", "x2");
title("True Marginal PDF")
savefig("figs_grid/truePDF.png")

## 
function plotErr(figNum, x1Fine, x2Fine, RHOPred, RHOTrue)
    XX = similar(RHOPred);
    YY = similar(RHOPred);

    for i = 1:length(x1Fine), j = 1:length(x2Fine)
        XX[i, j] = x1Fine[i]
        YY[i, j] = x2Fine[j]
    end

    figure(figNum, (12,4)); clf();
    subplot(1,3,1); 
    pcolor(XX, YY, RHOPred, shading = "auto", cmap = "inferno"); colorbar();    
    xlabel(L"x_1"); ylabel(L"x_2"); axis("auto")
    title("Predicted PDF");

    subplot(1,3,2); 
    pcolor(XX, YY, RHOTrue, shading = "auto", cmap = "inferno"); colorbar();    
    xlabel(L"x_1"); ylabel(L"x_2"); axis("auto")
    title("True PDF ");

    subplot(1,3,3); 
    absErr =  abs2.(RHOPred - RHOTrue);
    maxErr = @sprintf "%.2e" mean(absErr)
    pcolor(XX, YY, absErr, shading = "auto", cmap = "inferno"); colorbar();    
    xlabel(L"x_1"); ylabel(L"x_2"); axis("auto")
    title(L"$ ϵ_{\rm mse} $ = %$(maxErr)");

    tight_layout()
end
plotErr(27, x1Fine, x2Fine, RHO12FineP, RHO12FineT2);
savefig("figs_grid/exp$(expNum)/err.png");