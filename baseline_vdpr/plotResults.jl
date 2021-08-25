## Plot the results of the bsaeline-PINNs implementation for the Van der Pol Rayleigh oscillator
using JLD2, PyPlot, VectorizedRoutines, NeuralPDE, ModelingToolkit, LinearAlgebra, Flux, Trapz, Printf
@variables x1,x2
pygui(true);


# Load data 
activFunc = tanh; 
suff = string(activFunc); 
nn = 48; 
Q = 0.3;
rhoTrue(x) = exp(1/Q*(x[1]^2 + x[2]^2 - 1/2*(x[1]^2 + x[2]^2)^2));

cd(@__DIR__);
fileLoc = "data/dx5eM2_vdpr_$(suff)_$(nn)_2.jld2";

println("Loading file");
file =  jldopen(fileLoc, "r");
optParam = read(file, "optParam");
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
close(file);

parameterless_type_θ = DiffEqBase.parameterless_type(optParam);

chain = Chain(Dense(2,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));
parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ );

maxval = 2.0f0;

nEvalFine = 100;
xs = range(-maxval, maxval, length = nEvalFine);
ys = range(-maxval, maxval, length = nEvalFine)
rho_pred = Float32.([exp(first(phi([x,y],optParam))) for x in xs, y in ys]);
rho_true = Float32.([rhoTrue([x,y]) for x in xs, y in ys]);
##
rho_pred_norm = rho_pred/trapz((xs,ys),rho_pred);
rho_true_norm = rho_true/trapz((xs,ys),rho_true);

gridX, gridY = (VectorizedRoutines.Matlab.meshgrid(xs, ys));
rho_pred_norm = Float32.(reshape(rho_pred_norm, (length(ys), length(xs))));
rho_true_norm = Float32.(reshape(rho_true_norm, (length(ys), length(xs))));
gridX = Float32.(gridX); gridY = Float32.(gridY);
##
using PyPlot
pygui(true);
figure(1, (12,4)); clf();
subplot(1,3,1);
PyPlot.pcolor(gridX, gridY, rho_pred_norm, cmap="inferno", shading = "auto");
colorbar();
xlabel("x1"); ylabel("x2");
title("Prediction");
axis("auto");
tight_layout();

subplot(1,3,2);
PyPlot.pcolor(gridX, gridY, rho_true_norm, cmap="inferno",shading = "auto");
colorbar();xlabel("x1"); ylabel("x2");
title("Exact");
axis("auto");
tight_layout();

errNorm = abs.(rho_true_norm - rho_pred_norm);
mseRHOErr = sum(errNorm[:].^2)/length(errNorm);
mseRHOErrStr = @sprintf "%.2e" mseRHOErr;

subplot(1,3,3);
PyPlot.pcolor(gridX, gridY, errNorm, cmap="inferno",shading = "auto");
colorbar();xlabel("x1"); ylabel("x2");
title(L"Solution Error; $ϵ_{ρ}=$ %$mseRHOErrStr");
axis("auto");
tight_layout();



