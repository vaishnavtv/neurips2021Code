## Plot the results of the rothe-PINNs implementation for transient vdp
using JLD2,PyPlot,LinearAlgebra,Trapz,Printf,LaTeXStrings, Flux
pygui(true);
cd(@__DIR__);
include("../utils.jl")
mkpath("figs_rothe");

# Load data 
activFunc = tanh;
suff = string(activFunc);
nn = 100;
Q_fpke = 0.0f0#*1.0I(2); # σ^2

expNum = 35; 
fileLoc = "data_rothe/vdp_exp$(expNum).jld2";
@info "Loading file from ts_rothe exp $(expNum)"
file = jldopen(fileLoc, "r");
optParams = read(file, "optParams");
tR = read(file, "tR");
# dt_sim = 0.01;
# tR = LinRange(0.0,2, Int(0.1/dt_sim) + 1);
close(file);
println("Are any of the parameters NaN ever? $(sum([any(isnan.(optParams[i])) for i in 1:length(optParams)]))")

mkpath("figs_rothe/exp$(expNum)")


chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1, abs2));
# chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));

Q_fpke_str = string(Q_fpke);

## Neural network
initθ,re  = Flux.destructure(chain)
phi = (x,θ) -> re(θ)(Array(x))

## Domain
maxval = 4.0;
xmin = maxval * [-1, -1];
xmax = maxval * [1, 1];
NN = 100;#Int(sqrt(size(C,2)));
CEval = GenerateNDGrid(xmin, xmax, [1, 1] * NN);
XX = reshape(CEval[1, :], NN, NN);
YY = reshape(CEval[2, :], NN, NN);

dt = 0.5; tEnd = 5.0; 
tR_plot = LinRange(0.0, tEnd, Int(tEnd/dt)+1);
# dt_sim = 0.001;
# tR = LinRange(0.0, tEnd, Int(tEnd/dt_sim) + 1);
indC = findall(in(tR_plot*100), tR*100)
for (tInt, tVal) in enumerate(indC)
    println("Computing ρ on fine grid for t = $(tR[tVal])");
    # RHO = reshape(exp.(phi(CEval, optParams[tVal])),NN, NN); # approximating η
    RHO = reshape((phi(CEval, optParams[tVal])),NN, NN); # approximating ρ directly 
    RHO = RHO/trapz((XX[:,1], XX[:,1]), RHO);

    figure(48); clf();
    pcolormesh(XX, YY, RHO , cmap = "inferno", shading="auto"); colorbar();
    xlabel("x1"); ylabel("x2");
    title("ρ at t = $(tR[tVal]); Q = $(Q_fpke)")
    tight_layout();
    savefig("figs_rothe/exp$(expNum)/rho_t$(tInt).png")
    pause(0.001);
end

