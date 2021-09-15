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
dx = 0.01;
suff = string(activFunc);
nn = 20;
expNum = 11;
strategy = "grid";

Q_fpke = 0.1f0*1.0I(2); # σ^2

cd(@__DIR__);
include("missileDynamics.jl")
fileLoc = "data/ll_$(strategy)_missile_$(suff)_$(nn)_exp$(expNum).jld2";

println("Loading file");
file = jldopen(fileLoc, "r");
optParam = read(file, "optParam");
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
close(file);
println("Are any of the parameters NaN? $(any(isnan.(optParam)))")
## plot losses
nIters = length(PDE_losses);
figure(1); clf();
semilogy(1:nIters, PDE_losses, label =  "PDE");
semilogy(1:nIters, BC_losses, label = "BC");
xlabel("Iterations");
ylabel("ϵ");
title(string(strategy," Exp $(expNum)"));
legend();
tight_layout();
## Neural network
chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
# chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

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

    pdeErrFn(x) = (tr(df(x)) + dot(f(x), dη(x)) - 1/2*sum(Q_fpke.*(d2η(x) + dη(x)*dη(x)')))#
    return ρNetS, pdeErrFn
end
ρFn, pdeErrFn = ρ_pdeErr_fns(optParam);

# Domain
minM = 1.2; maxM = 2.5;
minα = -1.0; maxα = 1.5;
xxFine = range(minM, maxM, length = nEvalFine);
yyFine = range(minα, maxα, length = nEvalFine);

RHOPred = [ρFn([x, y]) for x in xxFine, y in yyFine];
FFFine = [pdeErrFn([x, y])^2 for x in xxFine, y in yyFine];
@show maximum(RHOPred)
maximum(FFFine)
ρFn(xTrim[ii])
@show ρFn([1,0])
# normalize
normC = trapz((xxFine, yyFine), RHOPred)
RHOFine = RHOPred / normC;

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
    pcolor(XXFine, YYFine, RHOPred, shading = "auto", cmap = "inferno")
    colorbar()
    xlabel("M")
    ylabel("α (rad)")
    axis("auto")
    title("Steady-State Solution (ρ)")

    subplot(1, 2, 2)
    pcolormesh(XXFine, YYFine, FFFine, cmap = "inferno", shading = "auto")
    colorbar()
    axis("auto")
    title(L"Equation Error; $ϵ_{pde}$ = %$(mseEqErrStr)")
    xlabel("M")
    ylabel("α (rad)")

    tight_layout()

end
plotDistErr(3);
# savefig("figs/quasi_exp$(expNum).png");

## normalisation as quadrature problem  
using Quadrature
ρFnQuad(z,p) = ρFn(z)#/normC
prob = QuadratureProblem(ρFnQuad, [minM, minα], [maxM, maxα], p = 0);
sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
@show sol.u
@show abs(sol.u - normC)