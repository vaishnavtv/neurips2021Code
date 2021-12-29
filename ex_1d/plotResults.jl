## Plot results for the 1D example
cd(@__DIR__);
using JLD2, NeuralPDE, Flux, Trapz, PyPlot, DiffEqBase
pygui(true);

fileLoc = "data/eta_exp$(expNum).jld2";
@info "Loading ss results for 1d system from exp $(expNum)";

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
xlabel("Iterations"); ylabel("ϵ");
title("Loss Function exp$(expNum)"); legend();
tight_layout();

chain = Chain(Dense(2,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));
parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

maxval = 2.2; dx = 0.01;
xs = -maxval:dx:maxval;
u_predict  = [exp(first(phi([x],optParam))) for x in xs]
norm_pred = trapz(xs, u_predict);
u_predict /= norm_pred; # normalized

ρ_true(x) = exp((1/(2*Q_fpke))*(2*α*x^2 - β*x^4)); # true analytical solution, before normalization
u_real = [ρ_true(x) for x in xs];
norm_real = trapz(xs, u_real);
u_real /= norm_real;

figure(1); clf();
plot(xs ,u_predict, label = "predict");
scatter(xs ,u_real, s = 0.5, c = "r", label = "true");
xlabel("x"); ylabel("ρ");
title("Steady-state Solution"); legend();
tight_layout();
if runExp 
    savefig("figs_eta/exp$(expNum).png");
end

figure(2); clf();
nIters = length(PDE_losses);
semilogy(1:nIters, PDE_losses, label = "PDE");
semilogy(1:nIters, BC_losses, label = "BC");
title("training loss");
xlabel("Iteration"); ylabel("ϵ")
tight_layout();