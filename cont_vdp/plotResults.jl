## Plot the results of the bsaeline-PINNs implementation for the Van der Pol Rayleigh oscillator
using JLD2, Plots, NeuralPDE, ModelingToolkit, LinearAlgebra, Flux, Trapz, Printf, DiffEqFlux
@variables x1,x2
gr();
# pygui(true);


# Load data 
activFunc = tanh; 
suff = string(activFunc); 
nn = 48; 
Q = 0.3;

# parameters for rhoSS_desired
μ_ss = zeros(2); Σ_ss = 0.001*1.0I(2);
rhoTrue(x) = exp(-1/2*(x - μ_ss)'*inv(Σ_ss)*(x - μ_ss))/(2*pi*sqrt(det(Σ_ss))); # desired steady-state distribution (gaussian function) 

cd(@__DIR__);
fileLoc = "data/dx5eM2_vdp_$(suff)_$(nn)_cont.jld2";

println("Loading file");
file =  jldopen(fileLoc, "r");
optParam = read(file, "optParam");
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
rhoSS_losses = read(file, "rhoSS_losses");
close(file);

parameterless_type_θ = DiffEqBase.parameterless_type(optParam);

chain1 = Chain(Dense(2,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));
chain2 = Chain(Dense(2,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));
chain = [chain1; chain2];
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain));
phi = NeuralPDE.get_phi.(chain, parameterless_type_θ);

maxval = 2.0f0;

nEvalFine = 100;
len1 = Int(length(optParam)/2);
optParam1 = optParam[1:len1];
optParam2 = optParam[len1+1:end];

xs = range(-maxval, maxval, length = nEvalFine);
ys = range(-maxval, maxval, length = nEvalFine)
rho_pred = Float32.([exp(first(phi[1]([x,y],optParam1))) for x in xs, y in ys]);
rho_true = Float32.([rhoTrue([x,y]) for x in xs, y in ys]);
##
rho_pred_norm = rho_pred/trapz((xs,ys),rho_pred);
rho_true_norm = rho_true/trapz((xs,ys),rho_true);

# gridX, gridY = (VectorizedRoutines.Matlab.meshgrid(xs, ys));
gridX = xs'.*ones(nEvalFine);
gridY = ones(nEvalFine)'.*ys;

rho_pred_norm = Float32.(reshape(rho_pred_norm, (length(ys), length(xs))));
rho_true_norm = Float32.(reshape(rho_true_norm, (length(ys), length(xs))));
gridX = Float32.(gridX); gridY = Float32.(gridY);
##
using Plots; gr();
# pygui(true);
# figure(1, (12,4)); clf();
# subplot(1,3,1);
p1 = contourf(xs,ys,rho_pred_norm);#, cmap="inferno", shading = "auto");
p2 = contourf(xs, ys, rho_true_norm);
# colorbar();
# xlabel("x1"); ylabel("x2");
# title("Prediction");
# axis("auto");
# tight_layout();

# subplot(1,3,2);
# PyPlot.pcolor(gridX, gridY, rho_true_norm, cmap="inferno",shading = "auto");
# colorbar();xlabel("x1"); ylabel("x2");
# title("Exact");
# axis("auto");
# tight_layout();

errNorm = abs.(rho_true_norm - rho_pred_norm);
mseRHOErr = sum(errNorm[:].^2)/length(errNorm);
mseRHOErrStr = @sprintf "%.2e" mseRHOErr;

p3 = contourf(xs, ys, errNorm);
plot(p1, p2, p3, layout = (1,3));
# subplot(1,3,3);
# PyPlot.pcolor(gridX, gridY, errNorm, cmap="inferno",shading = "auto");
# colorbar();xlabel("x1"); ylabel("x2");
# title(L"Solution Error; $ϵ_{ρ}=$ %$mseRHOErrStr");
# axis("auto");
# tight_layout();



