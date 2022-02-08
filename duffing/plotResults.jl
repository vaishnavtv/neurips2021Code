## Plot the results of the baseline-PINNs implementation for the Duffing oscillator
using JLD2, PyPlot, NeuralPDE, ModelingToolkit, LinearAlgebra, Flux, Trapz, Printf, LaTeXStrings, ForwardDiff, DiffEqFlux
pygui(true);

# Load data 
activFunc = tanh;
dx = 0.05;
suff = string(activFunc);
nn = 48;
optFlag = 1;
expNum = 9;
Q_fpke = 1f0;
strat = "grid";

cd(@__DIR__);
fileLoc = "data_$(strat)/ll_$(strat)_duff_exp$(expNum).jld2";

println("Loading file");
file = jldopen(fileLoc, "r");
optParam = read(file, "optParam");
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
# NORM_losses = read(file, "NORM_losses")
close(file);
println("Are any of the parameters NaN? $(any(isnan.(optParam)))")

## plot losses
nIters = length(PDE_losses);
figure(1); clf();
semilogy(1:nIters, PDE_losses, label =  "PDE");
semilogy(1:nIters, BC_losses, label = "BC");
# semilogy(1:nIters, NORM_losses, label = "NORM");
xlabel("Iterations");
ylabel("ϵ");
title("Loss Function $(strat) exp$(expNum)");
legend();
tight_layout();

##
# Neural network
chain = Chain(Dense(2,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));
# chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
# nn = 20; chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

# Duffing oscillator Dynamics
# η_duff = 0.2; α_duff = 1.0; β_duff = 0.2;
η_duff = 10f0; α_duff = -15f0; β_duff = 30f0;
f(x) = [x[2]; -η_duff.*x[2] .- α_duff.*x[1] .- β_duff.*x[1].^3];

function g(x::Vector)
    return [0.0f0;1.0f0];
end # diffusion

## set up error function
@parameters x1, x2
@variables η(..)

xSym = [x1; x2]
# PDE
ρ(x) = exp(η(x...));
#  PDE written directly in η
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G = diffC*η(xSym...);

T1 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(η(xSym...)))) for i in 1:length(xSym)]); # drift term
T2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
T2 += sum([(Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j]) - (Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])*η(xSym...) + diffC[i,j]*(Differential(xSym[i])(η(xSym...)))*(Differential(xSym[j])(η(xSym...))) for i in 1:length(xSym), j=1:length(xSym)]); # complete diffusion term

Eqn = expand_derivatives(-T1+T2); 
pdeOrig = simplify(Eqn, expand = true) ~ 0.0f0;
pde = pdeOrig;

initθ = DiffEqFlux.initial_params(chain) #|> gpu;
strategy = NeuralPDE.GridTraining(0.1);
derivative = NeuralPDE.get_numeric_derivative();

indvars = xSym
depvars = [η(xSym...)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

_pde_loss_function = NeuralPDE.build_loss_function(pde, indvars, depvars, phi, derivative, integral, chain, initθ, strategy);

maxval = 2.0f0;
nEvalFine = 100;

## True solution
rhoTrue(x) = exp(-η_duff / Q_fpke * (x[2].^2 + α_duff.*x[1].^2 + β_duff/2*x[1].^4));
# df(x) = ForwardDiff.jacobian(f,x);

# Generate functions to check solution and PDE error
function ρ_pdeErr_fns(optParam)
    function ηNetS(x)
        return first(phi(x, optParam))
    end
    ρNetS(x) = exp(ηNetS(x)) # solution after first iteration
    pdeErrFn(x) = first(_pde_loss_function(x, optParam))
    
    ## checking for the true solution 
    # ηNetS(x) = (-η_duff / Q_fpke * (x[2].^2 + α_duff.*x[1].^2 + β_duff/2*x[1].^4));
    # ρNetS(x) = rhoTrue(x);
    
    # dη(x) = ForwardDiff.gradient(ηNetS, x)
    # d2η(x) = ForwardDiff.jacobian(dη, x)
    # pdeErrFn(x) = tr(df(x)) + dot(f(x), dη(x)) - Q_fpke / 2 * (d2η(x)[end] + (dη(x)[end])^2)
    
    return ρNetS, pdeErrFn
end
ρFn, pdeErrFn = ρ_pdeErr_fns(optParam);

xxFine = range(-maxval, maxval, length = nEvalFine);
yyFine = range(-maxval, maxval, length = nEvalFine);

RHOPred = [ρFn([x, y]) for x in xxFine, y in yyFine];
RHOTrue = [rhoTrue([x, y]) for x in xxFine, y in yyFine];
FFFine = [pdeErrFn([x, y])^2 for x in xxFine, y in yyFine];

# normalize
RHOFine = RHOPred / trapz((xxFine, yyFine), RHOPred);
RHOTrueFine = RHOTrue/trapz((xxFine, yyFine), RHOTrue);

println("The mean squared equation error with dx=$(dx) is:")
mseEqErr = sum(FFFine[:] .^ 2) / length(FFFine);
@show mseEqErr;
mseEqErrStr = @sprintf "%.2e" mseEqErr;
XXFine = similar(RHOFine);
YYFine = similar(RHOFine);

for i = 1:nEvalFine, j = 1:nEvalFine
    XXFine[i, j] = xxFine[i]
    YYFine[i, j] = yyFine[j]
end

## Plot shown in paper
function plotDistErr(figNum)

    figure(figNum, [8, 8])
    clf()
    subplot(2, 2, 1)
    pcolor(XXFine, YYFine, RHOFine, shading = "auto", cmap = "jet"); colorbar()
    xlabel("x1")
    ylabel("x2")
    axis("auto")
    PyPlot.title("Prediction")

    subplot(2, 2, 2)
    pcolor(XXFine, YYFine, RHOTrueFine, shading = "auto", cmap = "jet");
    colorbar()
    xlabel("x1")
    ylabel("x2")
    axis("auto")
    PyPlot.title("Exact Solution")

    errNorm = abs.(RHOTrueFine - RHOFine);
    mseRHOErr = sum(errNorm[:] .^ 2) / length(errNorm);
    mseRHOErrStr = @sprintf "%.2e" mseRHOErr;

    subplot(2, 2, 3)
    pcolor(XXFine, YYFine, errNorm, shading = "auto", cmap = "inferno")
    colorbar()
    axis("auto")
    title(L"Solution Error; $ϵ_{ρ}=$ %$mseRHOErrStr");
    xlabel("x1")
    ylabel("x2")

    subplot(2, 2, 4)
    pcolor(XXFine, YYFine, FFFine, shading = "auto", cmap = "inferno")
    colorbar()
    axis("auto")
    title(L"Equation Error; $ϵ_{pde}$ = %$(mseEqErrStr)")
    xlabel("x1")
    ylabel("x2")

    tight_layout()

end
mkpath("figs_$(strat)")
plotDistErr(100+expNum);
savefig("figs_$(strat)/ss_duff_$(strat)_$(expNum).png")
