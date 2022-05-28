using JLD2,PyPlot,LinearAlgebra,Trapz,Printf,LaTeXStrings, Flux, Distributions
pygui(true);
cd(@__DIR__);
include("../utils.jl")
include("f18Dyn.jl")
mkpath("figs_rhoFixed")

## Load data 
activFunc = tanh;
suff = string(activFunc);
nn = 100;
Q_fpke = 0.0f0#*1.0I(2); # σ^2
tEnd = 500.0; dt = 2.0;

expNum = 11; 
fileLoc = "data_rhoConst/exp$(expNum).jld2";
@info "Loading file from ss_cont_f18_rhoFixed exp $(expNum)"
file = jldopen(fileLoc, "r");
optParam = Array(read(file, "optParam"));
PDE_losses = read(file, "PDE_losses");
close(file);
println("Are any of the parameters NaN ever? $(any(isnan.(optParam)))")

mkpath("figs_rhoFixed/exp$(expNum)")

## Plot Losses
nIters = length(PDE_losses);
figure(1); clf();
semilogy(1:nIters, PDE_losses, label =  "PDE");
xlabel("Iterations"); ylabel("ϵ");
title("Loss Function exp$(expNum)");
legend(); tight_layout();
savefig("figs_rhoFixed/exp$(expNum)/loss.png")

## Neural network set up
dim = 4 # number of dimensions
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
# chain2 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));

initθ,re  = Flux.destructure(chain)
phi = (x,θ) -> re(θ)(Array(x))

th1 = optParam[1:Int(length(optParam)/2)]; 
th2 = optParam[Int(length(optParam)/2) + 1:end];

## F18 Dynamics with controller Kc
using DifferentialEquations
f18_xTrim = Float32.(1.0e+02*[3.500000000000000;-0.000000000000000;0.003540971009312;-0.000189987747098;0.000322083113778;0.000459982356948;0.006108652381980;0.003262466855376;0;]);
indX = [1;3;8;5];

f18_uTrim = Float32.(1.0e+04 *[-0.000000767698465;-0.000002371697733;-0.000007859275313;1.449999997030301;]);
indU = [3;4];

maskIndx = zeros(Float32,(length(f18_xTrim),length(indX)));
maskIndu = zeros(Float32,(length(f18_uTrim),length(indU)));
for i in 1:length(indX)
    maskIndx[indX[i],i] = 1f0;
    if i<=length(indU)
        maskIndu[indU[i],i] = 1f0;
    end
end

function f18RedDyn(xd,t)
    # reduced order f18 dynamics
    ud = [first(phi(xd, th1)); first(phi(xd, th2))]
    
    tx = maskIndx*xd; tu = maskIndu*ud;
    xFull = f18_xTrim + maskIndx*xd; 
    uFull = [1f0;1f0;0f0;1f0].*f18_uTrim + maskIndu*ud;
    # uFull = f18_uTrim #+ maskIndu*ud;
 
    
    xdotFull = f18Dyn(xFull, uFull) #- f18Dyn(f18_xTrim, f18_uTrim)
    # xdotFull = f18Dyn(f18_xTrim, f18_uTrim)
    # @show maximum(xdotFull[indX])
    return (xdotFull[indX])
end

function nlSim(x0)
    # function to simulate nonlinear controlled dynamics with initial condition x0 and controller K
    odeFn(x, p, t) = f18RedDyn(x, t)
    prob = ODEProblem(odeFn, x0, (0.0, tEnd))
    sol = solve(prob, Tsit5(), saveat = dt, reltol = 1e-6, abstol = 1e-6)
    return sol
end
#
function plotTraj(sol, figNum)
    tSim = sol.t
    x1Sol = sol[1, :] #.+ f18_xTrim[indX[1]]
    x2Sol = sol[2, :] #.+ f18_xTrim[indX[2]]
    x3Sol = sol[3, :] #.+ f18_xTrim[indX[3]]
    x4Sol = sol[4, :] #.+ f18_xTrim[indX[4]]
    # u = [(1-exp(-τ*tSim[i]))*first(phi(sol.u[i], optParam)) for i = 1:length(tSim)]
    u1 = [first(phi(sol.u[i], th1)) for i = 1:length(tSim)];
    u2 = [first(phi(sol.u[i], th2)) for i = 1:length(tSim)];

    figure(figNum, figsize = (8,4));clf()
    subplot(3, 2, 1);
    PyPlot.plot(tSim, x1Sol);
    xlabel("t"); ylabel("V (ft/s)"); grid("on")

    subplot(3, 2, 2)
    PyPlot.plot(tSim, rad2deg.(x2Sol))
    xlabel("t");  ylabel("α (deg)"); grid("on")

    subplot(3, 2, 3)
    PyPlot.plot(tSim, rad2deg.(x3Sol))
    xlabel("t"); ylabel("θ (deg)"); grid("on")

    subplot(3, 2, 4);
    PyPlot.plot(tSim, rad2deg.(x4Sol));
    xlabel("t"); ylabel("q (deg/s)"); grid("on")

    subplot(3, 2, 5)
    PyPlot.plot(tSim, u1)
    xlabel("t");  ylabel(L"δ_{stab} ({\rm deg})"); grid("on")

    subplot(3, 2, 6)
    PyPlot.plot(tSim, u2)
    xlabel("t"); ylabel("T"); grid("on")

    suptitle("Exp$(expNum) Traj: Deviation from Trim");
    tight_layout()
end

xmin = 0.1*[-100f0;deg2rad(-10f0);deg2rad(-10f0); deg2rad(-5f0)];
xmax = -xmin;
tx = xmin .+ ((xmax - xmin) .* rand(4));
@show tx
solSim = nlSim(tx);
println("x1 initial value: $(solSim[1,1]);  x1 terminal value: $(solSim[1,end])");
println("x2 initial value: $(solSim[2,1]);  x2 terminal value: $(solSim[2,end])");
println("x3 initial value: $(solSim[3,1]);  x3 terminal value: $(solSim[3,end])");
println("x4 initial value: $(solSim[4,1]);  x4 terminal value: $(solSim[4,end])");
println("Terminal value state norm: $(norm(solSim[:,end]))");
# figure(4); clf();
plotTraj(solSim, 3);
# savefig("figs_rhoFixed/exp$(expNum)/traj.png")