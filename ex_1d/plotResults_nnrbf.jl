## Plot results for the 1D example using nnrbf
cd(@__DIR__);
include("../rb_nnrbf/nnrbf.jl");
include("../rb_nnrbf/libFPKE.jl");
using JLD2, NeuralPDE, Flux, Trapz, PyPlot
pygui(true);

expNum = 2;
nBasis = 5; # Number of basis functions in nnrbf
activFunc = tanh;
Q_fpke = 0.25;
α = 0.3f0; β = 0.5f0;
fileLoc = "data_nnrbf/eta_exp$(expNum).jld2";

@info "Loading ss results for 1d system from exp $(expNum)";

file = jldopen(fileLoc, "r");
optParam = read(file, "optParam");
# PDE_losses = read(file, "PDE_losses");
# BC_losses = read(file, "BC_losses");
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

xc = Float32.(0.1f0*randn(nBasis));
chain = Chain(Parallel(vcat,[NNRBF([1.0f0;;],[xc[i]],[-1.0f0;;],[xc[i]]) for i = 1:nBasis]), Linear(ones(Float32,(1,nBasis))));

parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

maxval = 2.2; dx = 0.01;
xs = -maxval:dx:maxval;
u_predict  = [(first(phi([x],optParam))) for x in xs]
norm_pred = trapz(xs, u_predict);
u_predict /= norm_pred; # normalized

ρ_true(x) = exp((1/(2*Q_fpke))*(2*α*x^2 - β*x^4)); # true analytical solution, before normalization
u_real = [ρ_true(x) for x in xs];
norm_real = trapz(xs, u_real);
u_real /= norm_real;

println("The maximum pointwise absolute error is $(maximum(abs.(u_predict - u_real)))");
##
figure(2, (8,4)); clf();
subplot(1,2,1);
plot(xs ,u_predict, label = "predict");
scatter(xs ,u_real, s = 0.5, c = "r", label = "true");
xlabel("x"); ylabel("ρ");
title("Exp$(expNum): Steady-state ρ"); legend();

subplot(1,2,2);
plot(xs, abs.(u_predict - u_real)); 
xlabel("x"); ylabel("ϵ");
title("Absolute Error");
tight_layout();
# savefig("figs_nnrbf/exp$(expNum).png");