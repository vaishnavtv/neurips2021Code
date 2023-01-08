## Plot the results of the rothe-PINNs implementation for transient vdp
using JLD2,PyPlot,LinearAlgebra,Trapz,Printf,LaTeXStrings, Flux, Distributions
pygui(true);
cd(@__DIR__);
include("../utils.jl")
mkpath("figs_def");

## Load data 
activFunc = tanh;
suff = string(activFunc);
nn = 48;
Q_fpke = 0.1f0#*1.0I(2); # σ^2

tEnd = 20.0f0;
dt = tEnd/10f0;

# expNum = 25; 
# fileLoc = "../ts_cont_vdp/data_cont_rothe/vdp_exp$(expNum).jld2";
# @info "Loading file from ts_cont_vdp exp $(expNum)"
# lab_plot = "ts";

# expNum = 9; 
# fileLoc = "data_rhoConst/exp$(expNum).jld2";
# @info "Loading file from ss2_cont_vdp exp $(expNum)"
# lab_plot = "ss2";

expNum = 9; 
fileLoc = "data/ss_cont_vdp_exp$(expNum).jld2";
@info "Loading file from ss1_cont_vdp exp $(expNum)"
lab_plot = "ss1";

file = jldopen(fileLoc, "r");
optParam = Array(read(file, "optParam"));
close(file);
println("Are any of the parameters NaN ever? $(any(isnan.(optParam)))")

if lab_plot == "ss1"
    len1 = Int(length(optParam) / 2);
    optParam1 = optParam[1:len1];
    optParam2 = optParam[len1+1:end];
    optParam = optParam2;
end

## Neural network
# chain = Chain(Dense(2, nn, activFunc), Dense(nn, 1)); # 1 layer network
chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1)); # 2 layer network
initθ,re  = Flux.destructure(chain)
phi = (x,θ) -> re(θ)(Array(x))


## controlled Dynamics do not stabilize x1
using DifferentialEquations
# Dynamics with controller Kc
function vdpDyn(x,t)
    dx = similar(x)
    u = first(phi(x, optParam))
    dx[1] = x[2]
    dx[2] = -x[1] + (1 - x[1]^2) * x[2] + u# (1-exp(-τ*t))*u
    return (dx)
end
#
function nlSim(x0)
    # function to simulate nonlinear controlled dynamics with initial condition x0 and controller K
    odeFn(x, p, t) = vdpDyn(x, t)
    prob = ODEProblem(odeFn, x0, (0.0, tEnd))
    sol = solve(prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)
    return sol
end

function plotTraj(sol, figNum)
    tSim = sol.t
    x1Sol = sol[1, :]
    x2Sol = sol[2, :]
    # u = [(1-exp(-τ*tSim[i]))*first(phi(sol.u[i], optParam)) for i = 1:length(tSim)]
    u = [first(phi(sol.u[i], optParam)) for i = 1:length(tSim)]

    figure(figNum);#clf()
    subplot(3, 1, 1);
    PyPlot.plot(tSim, x1Sol, label = lab_plot);
    xlabel("t"); ylabel(L"x_1")
    grid("on")
    legend()

    subplot(3, 1, 2)
    PyPlot.plot(tSim, x2Sol, label = lab_plot)
    xlabel("t");  ylabel(L"x_2")
    grid("on")
    legend()

    subplot(3, 1, 3)
    PyPlot.plot(tSim, u, label = lab_plot)
    xlabel("t"); ylabel("u")
    tight_layout()
    grid("on")
    legend()
    # suptitle("traj_$(titleSuff)");
    suptitle("Trajectory Comparison");
    tight_layout()
end

tx = -4.0 .+ 8 * rand(2);
@show tx
solSim = nlSim(tx);
println("x1 initial value: $(solSim[1,1]);  x1 terminal value: $(solSim[1,end])");
println("Terminal value state norm: $(norm(solSim[:,end]))");
plotTraj(solSim, 3);
savefig("figs_def/traj.png");
