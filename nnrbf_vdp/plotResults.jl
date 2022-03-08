using  JLD2, Symbolics, LinearAlgebra, Printf, Trapz, ForwardDiff, PyPlot
pygui(true);
cd(@__DIR__);
include("../../RadialBasisFunctions/rbfNet.jl")
include("../rb_nnrbf/libFPKE.jl")
mkpath("figs_nnrbf");

## Load data
println("Loading file");
expNum = 8;
fileLoc = "data_nnrbf/vdp_exp$(expNum).jld2";
Q_fpke = 0.1;

file = jldopen(fileLoc, "r");
XC = read(file, "XC");# basis centers
optParam = read(file, "optParam"); # parameters
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
NORM_losses = read(file, "NORM_losses");
close(file);

## plot losses
nIters = length(PDE_losses);
figure(1); clf();
semilogy(1:nIters, PDE_losses, label =  "PDE");
semilogy(1:nIters, BC_losses, label = "BC");
semilogy(1:nIters, NORM_losses, label = "NORM");
xlabel("Iterations"); ylabel("ϵ"); legend()
title("(NP) Loss Functions exp$(expNum)"); 
tight_layout();

## PDE
# Van der Pol Dynamics
d = 2;
nBasis = length(XC); nParams = 2*d^2 + 1;
f(x) = [x[2]; -x[1] + (1 - x[1]^2) * x[2]];
g(x) = [0.0; 1.0].*1.0I(2); # diffusion

# Get the FPKE error
pdeError, ∇pde_error, Eqn, ∇Eqn = ForwardFPKE_Ito(d;exponential=false); # FPKE
sysDers = SymbolicDerivativesOfDynamics(f,g,Q_fpke,d);

eqn_error(P,XC,x) = pdeError(ρ(P,XC,x), ρ_x(P,XC,x), ρ_xx(P,XC,x), sysDers[1](x), sysDers[2](x), sysDers[3](x), sysDers[4](x), sysDers[5](x));

p2P(p,nBasis,nParams) = [p[((i-1)*nParams+1):i*nParams] for i = 1:nBasis];

Popt = p2P(optParam, nBasis, nParams);

## PLOTS
maxval = 4.0;
xmin = maxval * [-1, -1];
xmax = maxval * [1, 1];
NN = 100;
CEval = GenerateNDGrid(xmin, xmax, [1, 1] * NN);
XX = reshape(CEval[1, :], NN, NN);
YY = reshape(CEval[2, :], NN, NN);

println("Computing ρ on fine grid");
RHO = reshape([ρ(Popt,XC,CEval[:,i]) for i in 1:size(CEval,2)],NN,:);
RHO = RHO/trapz((XX[:,1], XX[:,1]), RHO);

println("Computing error on fine grid:");
errFine = reshape([eqn_error(Popt,XC,CEval[:,i])^2 for i in 1:size(CEval,2)],NN,NN);
errFineNorm = norm(errFine, Inf); #ErrorXX(ff, C, CEval, β, α); # ∞ norm of error on fine grid
println("ϵ_{∞} on the fine grid is: $(errFineNorm)"); # infinity norm of error
##
errFineStr = @sprintf "%.2e" errFineNorm;
figure(90, (8,4)); clf();
subplot(1,2,1);
pcolormesh(XX, YY, RHO, cmap="inferno", shading="auto"); colorbar();
scatter([XC[i][1] for i in 1:nBasis], [XC[i][2] for i in 1:nBasis], s = 2, c = "w")
xlabel("x1"); ylabel("x2");
title("Steady-state solution ρ")
tight_layout();

subplot(1,2,2);
pcolormesh(XX, YY, errFine, cmap = "inferno", shading = "auto"); colorbar();
xlabel("x1"); ylabel("x2");
title("Equation Error Squared");
suptitle(string("(NP) nBasis = $(length(XC)); Q = $(Q_fpke); ", L"$ϵ_∞$ = ","$(errFineStr); ", "nIters = $(nIters-1)"))
# suptitle("(NP)nBasis = $(length(XC)); Q = $(Q_fpke); ϵ = $(errFineStr); nIters = $(nIters-1)");
tight_layout();
savefig("figs_nnrbf/exp$(expNum).png");

# 