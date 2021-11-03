## Plot the results of the baseline-PINNs implementation for the missile
include("cpu_missileDynamics.jl")

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
xSym = [x1, x2];

# Load data 
activFunc = tanh;
dx = 0.01;
suff = string(activFunc);
nn = 48;
expNum = 27;
strat = "quasi";
@info "Plotting results for missile exp$(expNum) using $(strat) strategy"

const dim = 2;
Q_fpke = 0.01f0#*1.0I(2); # σ^2
diffC = 0.5 * (g(xSym) * Q_fpke * g(xSym)'); # diffusion coefficient (constant in our case, not a fn of x)

cd(@__DIR__);
# fileLoc = "data_$(strat)/ll_$(strat)_missile_exp$(expNum).jld2";
fileLoc = "dataQuasi/ll_$(strat)_missile_$(suff)_$(nn)_exp$(expNum).jld2";

println("Loading file");
file = jldopen(fileLoc, "r");
optParam = read(file, "optParam");
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
# NORM_losses = read(file, "NORM_losses");
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
title(string(strat," Exp $(expNum)"));
legend();
tight_layout();
## Neural network
# chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
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
    dg(x) = ForwardDiff.jacobian(z->diag(g(z)),x);
    dρ(x) = ForwardDiff.gradient(ρNetS,x) 
    dη(x) = ForwardDiff.gradient(ηNetS, x)
    d2η(x) = ForwardDiff.jacobian(dη, x)

    # pdeErrFn(x) = (tr(df(x)) + dot(f(x), dη(x)) - sum(diffC.*(d2η(x) + dη(x)*dη(x)')))#

    D1(x) = f(x)*ρNetS(x) + 0.5f0*dg(x)*Q_fpke*diag(g(x));
    D2(x) = 0.5f0*g(x)*Q_fpke*g(x)'*ρNetS(x);
    dD1(x) = ForwardDiff.jacobian(D1,x);
    function diffTerm(x)
        out = 0;
        for j=1:dim
            for i=1:dim
                D2_ij(x) = D2(x)[i,j];
                out += ForwardDiff.hessian(D2_ij,x)[i,j];
            end
        end
        return out
    end
    pdeErrFn(x) = tr(dD1(x)) - diffTerm(x);

    # diffCTerm(x) = (0.5*g(x)*Q_fpke*g(x)');
    # diffC1(x) = first(diffCTerm(x));
    # pdeErrFn(x) = tr(df(x)) + dot(f(x), dη(x)) - (sum(diffCTerm(x).*(d2η(x) + dη(x)*dη(x)')) + first(ForwardDiff.hessian(diffC1, x))+ 2*first(dρ(x))*first(ForwardDiff.gradient(diffC1,x)))
    return ρNetS, pdeErrFn
end
ρFn, pdeErrFn = ρ_pdeErr_fns(optParam);

# Domain
minM = 0.8; maxM = 2.5;
minα = -1.0; maxα = 1.5;
xxFine = range(minM, maxM, length = nEvalFine);
yyFine = range(minα, maxα, length = nEvalFine);

RHOPred = [ρFn([x, y]) for x in xxFine, y in yyFine];
FFFine = [pdeErrFn([x, y])^2 for x in xxFine, y in yyFine]; # squared equation error on the evaluation grid
@show maximum(RHOPred)
@show maximum(FFFine)
@show ρFn(xTrim[ii])
# normalize
normC = trapz((xxFine, yyFine), RHOPred)
RHOFine = RHOPred / normC;

println("The mean squared equation error is:")
mseEqErr = sum(FFFine[:]) / length(FFFine);
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
    # suptitle("FT; g = g(x); Q_fpke = $(Q_fpke);")
    # suptitle("RT; g = $((g(xSym))); Q_fpke = $(Q_fpke);")
    tight_layout()

end
plotDistErr(3);
savefig("figs_$(strat)/$(strat)_exp$(expNum).png");

## normalisation as quadrature problem  
# using Quadrature
# ρFnQuad(z,p) = ρFn(z)#/normC
# prob = QuadratureProblem(ρFnQuad, [minM, minα], [maxM, maxα], p = 0);
# sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
# @show sol.u
# @show abs(sol.u - normC)