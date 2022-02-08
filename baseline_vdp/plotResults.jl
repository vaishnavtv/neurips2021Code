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
xSym = [x1;x2];
## rcParams
# rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
# rcParams["font.size"] = 15
# Load data 
activFunc = tanh;
dx = 0.05;
suff = string(activFunc);
nn = 100;
optFlag = 1;
expNum = 19;
Q_fpke = 0.1;
strat = "grid";

cd(@__DIR__);
fileLoc = "data_$(strat)/ll_$(strat)_vdp_exp$(expNum).jld2";
# fileLoc = "data/dx5eM2_vdp_tanh_48.jld2"
@info "Loading ss_vdp results from exp: $(expNum) using $(strat) strategy";
file = jldopen(fileLoc, "r");
# optParam = read(file, "p1");
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
title(string(strat," Exp $(expNum)"));
legend();
tight_layout();
##
# Neural network
# chain = Chain(Dense(2,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));
chain = Chain(Dense(2,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));
# chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

# Van der Pol Dynamics
f(x) = [x[2]; -x[1] + (1 - x[1]^2) * x[2]];

maxval = 4.0f0;

nEvalFine = 100;

# Generate functions to check solution and PDE error
# driftTerm = (Differential(x1)(x2*exp(η(x1, x2))) + Differential(x2)(exp(η(x1, x2))*(x2*(1 - (x1^2)) - x1)))*(exp(η(x1, x2))^-1)
# diffTerm1 = Differential(x2)(Differential(x2)(η(x1,x2))) 
# diffTerm2 = abs2(Differential(x2)(η(x1,x2))) # works
# diffTerm = Q_fpke/2*(diffTerm1 + diffTerm2); # diffusion term

# pde = driftTerm - diffTerm ~ 0.0f0 # full pde


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
maxPtWiseErr = maximum(FFFine);
@show maxPtWiseErr;

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
    pcolor(XXFine, YYFine, RHOFine, shading = "auto", cmap = "inferno");
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
plotDistErr(90+expNum);
savefig("figs_$(strat)/ss_vdp_$(strat)_exp$(expNum).png");

## quadrature error function
# using Quadrature, Cubature, Cuba
# errFnQuad(z,p) = (pdeErrFn(z))^2;
# prob = QuadratureProblem(errFnQuad, [-maxval, -maxval], [maxval, maxval], p = 0);
# sol = solve(prob,CubatureJLh(),reltol=1e-6,abstol=1e-3) 
# @show sol.u

## normalisation as quadrature problem  
# μ_ss = [-2.0f0,2.0f0];
# Σ_ss = Float32.(0.01 * 1.0I(2));
# rho0(x) = exp(-0.5f0 * (x - μ_ss)' * inv(Σ_ss) * (x - μ_ss)) / Float32(2 * pi * sqrt(det(Σ_ss)));
# rho0Pred = [rho0([x, y]) for x in xxFine, y in yyFine];
# normT0 = trapz((xxFine, yyFine), rho0Pred);
# @show normT0
# figure(97); clf();
# pcolor(XXFine, YYFine, rho0Pred);
# colorbar();
# tight_layout();

# using Quadrature, Cubature, Cuba
# ρFnQuad(z,p) = rho0(z)#/normC
# prob = QuadratureProblem(ρFnQuad, [-maxval, -maxval], [maxval, maxval], p = 0);
# # sol = solve(prob,HCubatureJL(),reltol=1e-6,abstol=1e-6) # wrong
# # sol = solve(prob,CubatureJLh(),reltol=1e-6,abstol=1e-6)# wrong
# # sol = solve(prob,CubatureJLp(),reltol=1e-6,abstol=1e-6)# wrong
# # sol = solve(prob,CubaCuhre(),reltol=1e-6,abstol=1e-6)# wrong
# # sol = solve(prob,CubaVegas(),reltol=1e-3,abstol=1e-3)# wrong
# sol = solve(prob,CubaSUAVE(),reltol=1e-3,abstol=1e-3)# correct
# sol = solve(prob,CubaDivonne(),reltol=1e-3,abstol=1e-3)# correct
# @show sol.u