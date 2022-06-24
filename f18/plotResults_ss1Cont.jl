# Solving for both ρ_ss and u
using JLD2,PyPlot,LinearAlgebra,Trapz,Printf,LaTeXStrings, Flux, Distributions
pygui(true);
cd(@__DIR__);
include("../utils.jl")
include("f18Dyn.jl")
mkpath("figs_ss1Cont")

import Random: seed!; seed!(1);

## Load data 
activFunc = tanh;
suff = string(activFunc);
nn = 100;
Q_fpke = 0.1f0#*1.0I(2); # σ^2
tEnd = 100.0; dt = 2.0;

nInp = 1;

expNum = 4; 
# fileLoc = "data_rhoConst/exp$(expNum).jld2";
fileLoc = "data_ss1_cont/exp$(expNum).jld2";
@info "Loading file from ss1_cont_f18 exp $(expNum)"
file = jldopen(fileLoc, "r");
optParam = Array(read(file, "optParam"));
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
rhoSS_losses = read(file, "rhoSS_losses");
close(file);
println("Are any of the parameters NaN ever? $(any(isnan.(optParam)))")

mkpath("figs_ss1Cont/exp$(expNum)")

## Plot Losses
nIters = length(PDE_losses);
figure(1); clf();
semilogy(1:nIters, PDE_losses, label =  "PDE");
semilogy(1:nIters, BC_losses, label =  "BC");
semilogy(1:nIters, rhoSS_losses, label =  "ρSS");
xlabel("Iterations"); ylabel("ϵ");
title("Loss Function exp$(expNum)");
legend(); tight_layout();
savefig("figs_ss1Cont/exp$(expNum)/loss.png")

## Neural network set up
dim = 4 # number of dimensions
# chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc),Dense(nn, nn, activFunc), Dense(nn, 1));
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));

initθ,re  = Flux.destructure(chain)
phi = (x,θ) -> re(θ)(Array(x))

if nInp == 1
    th1 = optParam[1:Int(length(optParam)/2)]; 
    th2 = optParam[Int(length(optParam)/2) + 1:end];
elseif nInp == 2
    # when rho, u1, u2
    th1 = optParam[1:Int(length(optParam)/3)]; 
    th2 = optParam[Int(length(optParam)/3) + 1:2*Int(length(optParam)/3)];
    th3 = optParam[2*Int(length(optParam)/3) + 1:end];
end

## F18 Dynamics with controller Kc
using DifferentialEquations
f18_xTrim = Float32.(1.0e+02*[3.500000000000000;-0.000000000000000;0.003540971009312;-0.000189987747098;0.000322083113778;0.000459982356948;0.006108652381980;0.003262466855376;0;]);
indX = [1;3;8;5];

f18_uTrim = Float32.(1.0e+04 *[-0.000000767698465;-0.000002371697733;-0.000007859275313;1.449999997030301;]);
if nInp == 1
    indU = [3]
elseif  nInp == 2
    indU = [3;4];
end

maskIndx = zeros(Float32,(length(f18_xTrim),length(indX)));
maskIndu = zeros(Float32,(length(f18_uTrim),length(indU)));
for i in 1:length(indX)
    maskIndx[indX[i],i] = 1f0;
    if i<=length(indU)
        maskIndu[indU[i],i] = 1f0;
    end
end
maskTrim = ones(Float32,length(f18_xTrim)); maskTrim[indX] .= 0f0; # keeping nonrelevant states constant
    
function f18RedDyn(xd,t)
    # reduced order f18 dynamics
    if nInp == 1
        ud = first(phi(xd, th2))
    elseif  nInp == 2
        ud = [first(phi(xd, th2)); first(phi(xd, th3))]
    end
    
    tx = maskIndx*xd; tu = maskIndu*ud;
    
    xFull = f18_xTrim + maskIndx*xd; 
    uFull = [1f0;1f0;0f0;1f0].*f18_uTrim + maskIndu*ud;
    # uFull = f18_uTrim + maskIndu*ud;

    # xFull = maskTrim.*f18_xTrim + maskIndx*xd; 
    # uFull = [1f0;1f0;0f0;0f0].*f18_uTrim + maskIndu*ud;
    
    xdotFull = f18Dyn(xFull, uFull) #- f18Dyn(f18_xTrim, f18_uTrim)
    return (xdotFull[indX])
end

function nlSim(x0)
    # function to simulate nonlinear controlled dynamics with initial condition x0 and controller K
    odeFn(x, p, t) = f18RedDyn(x, t)
    prob = ODEProblem(odeFn, x0, (0.0, tEnd))
    sol = solve(prob, Tsit5(), saveat = dt, reltol = 1e-6, abstol = 1e-6)
    return sol
end

function angBounds(x)
    # keep angle between -pi to pi
    y = mod(x,(2*pi)) + (-pi)
    return y
end

#
function plotTraj(sol, figNum)
    tSim = sol.t
    x1Sol = sol[1, :] #.- f18_xTrim[indX[1]]
    x2Sol = (sol[2, :]) #.- f18_xTrim[indX[2]]
    x3Sol = (sol[3, :]) #.- f18_xTrim[indX[3]]
    x4Sol = sol[4, :] #.- f18_xTrim[indX[4]]
    # u = [(1-exp(-τ*tSim[i]))*first(phi(sol.u[i], optParam)) for i = 1:length(tSim)]
    u1 = [first(phi(sol.u[i], th2)) for i = 1:length(tSim)]# .- f18_uTrim[indU[1]];
    if nInp == 2
        u2 = [first(phi(sol.u[i], th3)) for i = 1:length(tSim)]# .- f18_uTrim[indU[2]];
    end
    # u2 = 0f0*ones(length(u1)) #*[first(phi(sol.u[i], th2)) for i = 1:length(tSim)];

    figure(figNum, figsize = (8,4));#clf()
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
    PyPlot.plot(tSim, rad2deg.(u1))
    xlabel("t");  ylabel(L"δ_{stab} ({\rm deg})"); grid("on")

    if nInp == 2
        subplot(3, 2, 6)
        PyPlot.plot(tSim, u2)
        xlabel("t"); ylabel("T"); grid("on")
    end
    suptitle("Exp$(expNum) Traj: Deviation from x̄");
    tight_layout()
end
vmin = [-100f0;deg2rad(-10f0);deg2rad(-10f0); deg2rad(-5f0)] ;
xmin = 1f0*vmin #.+ f18_xTrim[indX]; 
xmax = 0f0*vmin #.+ f18_xTrim[indX]
tx = xmin .+ ((xmax - xmin) .* rand(4));
@show tx
solSim = nlSim(tx);
println("x1 initial value: $(solSim[1,1]);  x1 terminal value: $(solSim[1,end])");
println("x2 initial value: $(rad2deg(solSim[2,1])) deg;  x2 terminal value: $(rad2deg(solSim[2,end])) deg");
println("x3 initial value: $(rad2deg(solSim[3,1])) deg;  x3 terminal value: $(rad2deg(solSim[3,end])) deg");
println("x4 initial value: $(rad2deg(solSim[4,1])) deg/s;  x4 terminal value: $(rad2deg(solSim[4,end])) deg/s");
println("Terminal value state norm: $(norm(solSim[:,end]))");
figure(3); clf();
plotTraj(solSim, 3);
savefig("figs_ss1Cont/exp$(expNum)/traj.png")
# savefig("figs_rhoFixed/exp$(expNum)/traj.png")

## Generate 20 random trajectories
# seed!(1);
# nTraj = 10; figNum = nTraj;
# tx_nT = [xmin .+ ((xmax - xmin) .* rand(4)) for i in 1:nTraj];
# solSim_nT = [nlSim(txi) for txi in tx_nT];
# figure(figNum, figsize = (8,4)); clf();
# [plotTraj(solSim_nT[i], figNum) for i in 1:nTraj]; 


##
function fs(xd, phi, th1, th2)

    ud = [first(phi(xd, th1)); first(phi(xd, th2))]

    # perturbation about trim point
    xFull = f18_xTrim + maskIndx*xd; 
    # uFull = [1f0;1f0;0f0;0f0].*f18_uTrim + maskIndu*ud;
    uFull = f18_uTrim + maskIndu*ud;

    xdotFull = f18Dyn(xFull, uFull)
    # xdotFull = xFull;

    return (xdotFull[indX]) # return the 4 state dynamics

end

using ForwardDiff, Distributions
function tl(th1, th2, y, phi)
    μ_ss = [0f0,0f0,0f0,0f0] 
    Σ_ss = 0.1f0*Array(f18_xTrim[indX]).*1.0f0I(4)
    rho(x) =  pdf(MvNormal(μ_ss, Σ_ss),x);
    drho(x) = ForwardDiff.gradient(rho, x);

    dfs(x) = diag(ForwardDiff.jacobian(z->fs(z, phi, th1, th2), x));
    # dphi(x) = ForwardDiff.gradient(z->first(phi(z, th0)), x);
    @show dfs(y)
    @show drho(y)
    l = rho(y)*sum(dfs(y)) +  dot(fs(y, phi, th1, th2), drho(y));
    return l
end
# txEnd = solSim[:,end];
# @show tl(th1, th2, tx, phi)