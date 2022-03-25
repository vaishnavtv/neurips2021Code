## Plot the results of the baseline-PINNs implementation for the Van der Pol oscillator
using JLD2,PyPlot,ModelingToolkit,LinearAlgebra,Flux,Trapz,Printf,LaTeXStrings,ForwardDiff, DiffEqBase, Adapt#, NeuralPDE
pygui(true);
@variables x1, x2
xSym = [x1;x2];

# Load data 
activFunc = tanh;
dx = 0.05;
suff = string(activFunc);
nn = 28;
optFlag = 1;
expNum = 41;
Q_fpke = 0.1;
strat = "grid";

cd(@__DIR__);
fileLoc = "data_$(strat)/ll_$(strat)_vdp_exp$(expNum).jld2";
# fileLoc = "data/dx5eM2_vdp_tanh_48.jld2"
@info "Loading ss_vdp results from exp: $(expNum) using $(strat) strategy";
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
title("Loss Function exp$(expNum)");
title(string(strat," Exp $(expNum)"));
legend();
tight_layout();
##
# Neural network
# chain = Chain(Dense(2,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1)); # 4 HLs
# chain = Chain(Dense(2,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1)); # 3 HLs
chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1)); # 2 HLs
# parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
# phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
initθ,re  = Flux.destructure(chain)
phi = (x,θ) -> re(θ)(Array(x))

# Van der Pol Dynamics
f(x) = [x[2]; -x[1] + (1 - x[1]^2) * x[2]];
g(x) = [0.0f0;1.0f0];

maxval = 4.0f0;

nEvalFine = 100;

# Generate functions to check solution and PDE error
# driftTerm = (Differential(x1)(x2*exp(η(x1, x2))) + Differential(x2)(exp(η(x1, x2))*(x2*(1 - (x1^2)) - x1)))*(exp(η(x1, x2))^-1)
# diffTerm1 = Differential(x2)(Differential(x2)(η(x1,x2))) 
# diffTerm2 = abs2(Differential(x2)(η(x1,x2))) # works
# diffTerm = Q_fpke/2*(diffTerm1 + diffTerm2); # diffusion term

# pde = driftTerm - diffTerm ~ 0.0f0 # full pde

# PDE in ρ
# @parameters x1, x2
# @variables  η(..), ρs(..)
# xSym = [x1;x2];

# ρ(x) = (ρs(x[1],x[2]));
# F = f(xSym)*ρ(xSym);
# G = 0.5f0*(g(xSym)*Q_fpke*g(xSym)')*ρ(xSym);

# T1 = sum([Differential(xSym[i])(F[i]) for i in 1:length(xSym)]);
# T2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);

# Eqn = expand_derivatives(-T1+T2); # + dx*u(x1,x2)-1 ~ 0;
# pdeOrig = simplify(Eqn) ~ 0.0f0;
# pde = pdeOrig;

# # # # Setting up loss function using NeuralPDE API
# strategy = NeuralPDE.GridTraining(dx);
# derivative = NeuralPDE.get_numeric_derivative();

# indvars = [x1, x2]
# depvars = [ρs(x1,x2)] #[η(x1,x2)]

# integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

# _pde_loss_function = NeuralPDE.build_loss_function(pde,indvars,depvars,phi,derivative,integral,chain,initθ,strategy);

function ρ_pdeErr_fns(optParam)
    # ρNetS(x) = first(phi(x, optParam));
    # pdeErrFn(x) = first(_pde_loss_function(x, optParam))

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
plotDistErr(90);
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