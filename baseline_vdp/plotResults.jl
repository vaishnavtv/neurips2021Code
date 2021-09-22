## Plot the results of the baseline-PINNs implementation for the Van der Pol oscillator
using JLD2,
    PyPlot,
    NeuralPDE,
    ModelingToolkit,
    LinearAlgebra,
    Flux,
    Trapz,
    Printf,
    LaTeXStrings,
    ForwardDiff
pygui(true);
@variables x1, x2

# Load data 
activFunc = tanh;
dx = 0.05;
suff = string(activFunc);
nn = 48;
optFlag = 1;
expNum = 3;
Q = 0.1;
strat = "grid";

cd(@__DIR__);
fileLoc = "data_$(strat)/ll_$(strat)_vdp_exp$(expNum).jld2";

println("Loading file");
file = jldopen(fileLoc, "r");
optParam = read(file, "optParam");
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
close(file);
println("Are any of the parameters NaN? $(any(isnan.(optParam)))")

# Neural network
chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

# Van der Pol Dynamics
f(x) = [x[2]; -x[1] + (1 - x[1]^2) * x[2]];

maxval = 4.0f0;

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

    pdeErrFn(x) = tr(df(x)) + dot(f(x), dη(x)) - Q / 2 * (d2η(x)[end] + (dη(x)[end])^2)
    return ρNetS, pdeErrFn
end
ρFn, pdeErrFn = ρ_pdeErr_fns(optParam);

xxFine = range(-maxval, maxval, length = nEvalFine);
yyFine = range(-maxval, maxval, length = nEvalFine);

RHOFine = [ρFn([x, y]) for x in xxFine, y in yyFine];
FFFine = [pdeErrFn([x, y])^2 for x in xxFine, y in yyFine];

# normalize
RHOFine = RHOFine / trapz((xxFine, yyFine), RHOFine);

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
    pcolor(XXFine, YYFine, RHOFine, shading = "auto", cmap = "inferno")
    colorbar()
    xlabel("x1")
    ylabel("x2")
    axis("auto")
    title("Steady-State Solution (ρ)")

    subplot(1, 2, 2)
    pcolormesh(XXFine, YYFine, FFFine, cmap = "inferno", shading = "auto")
    colorbar()
    axis("auto")
    title(L"Equation Error; $ϵ_{pde}$ = %$(mseEqErrStr)")
    xlabel("x1")
    ylabel("x2")

    tight_layout()

end
plotDistErr(3);

