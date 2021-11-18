## Plot the results of the baseline-PINNs implementation for the Duffing oscillator
using JLD2,
    PyPlot,
    NeuralPDE,
    # ModelingToolkit,
    LinearAlgebra,
    Flux,
    Trapz,
    Printf,
    LaTeXStrings,
    ForwardDiff
pygui(true);
# @variables x1, x2

# Load data 
activFunc = tanh;
dx = 0.05;
suff = string(activFunc);
nn = 48;
optFlag = 1;
expNum = 1;
Q_fpke = 0.1;
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
title("Loss Function exp$(expNum)");
legend();
tight_layout();

##
# Neural network
chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

# Duffing oscillator Dynamics
η_duff = 0.2; α_duff = 1.0; β_duff = 0.2;
f(x) = [x[2]; η_duff.*x[2] .- α_duff.*x[1] .- β_duff.*x[1].^3];

maxval = 2.0f0;
nEvalFine = 100;

# Generate functions to check solution and PDE error
function ρ_pdeErr_fns(optParam)
    function ηNetS(x)
        return first(phi(x, optParam))
    end
    ρNetS(x) = exp(ηNetS(x)) # solution after first iteration
    df(x) = ForwardDiff.jacobian(f, x)
    dη(x) = ForwardDiff.gradient(ηNetS, x)
    d2η(x) = ForwardDiff.jacobian(dη, x)

    pdeErrFn(x) = tr(df(x)) + dot(f(x), dη(x)) - Q_fpke / 2 * (d2η(x)[end] + (dη(x)[end])^2)
    return ρNetS, pdeErrFn
end
ρFn, pdeErrFn = ρ_pdeErr_fns(optParam);

xxFine = range(-maxval, maxval, length = nEvalFine);
yyFine = range(-maxval, maxval, length = nEvalFine);

RHOPred = [ρFn([x, y]) for x in xxFine, y in yyFine];
FFFine = [pdeErrFn([x, y])^2 for x in xxFine, y in yyFine];

# normalize
RHOFine = RHOPred / trapz((xxFine, yyFine), RHOPred);

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

    figure(figNum, [8, 4])
    clf()
    subplot(1, 2, 1)
    # figure(figNum); clf();
    pcolor(XXFine, YYFine, RHOPred, shading = "auto", cmap = "inferno");
    colorbar()
    xlabel("x1")
    ylabel("x2")
    axis("auto")
    PyPlot.title("Steady-State Solution (ρ)")

    subplot(1, 2, 2)
    pcolor(XXFine, YYFine, FFFine, shading = "auto", cmap = "inferno")
    colorbar()
    axis("auto")
    PyPlot.title(L"Equation Error; $ϵ_{pde}$ = %$(mseEqErrStr)")
    xlabel("x1")
    ylabel("x2")

    tight_layout()

end
mkpath("figs_$(strat)")
plotDistErr(expNum);
savefig("figs_$(strat)/ss_duff_$(strat)_$(expNum).png")
