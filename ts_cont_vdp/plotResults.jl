## Plot the results of the rothe-PINNs implementation for transient vdp
using JLD2,PyPlot,LinearAlgebra,Trapz,Printf,LaTeXStrings, Flux, Distributions
pygui(true);
cd(@__DIR__);
include("../utils.jl")
mkpath("figs_cont");

#
# dt = 0.2; tEnd = 1.0;
# μ0  = [0f0,0f0]; Σ0 = 0.1f0*1.0f0I(2); #gaussian 
# nT = Int(tEnd/dt) + 1
# tR = LinRange(0.0, tEnd,Int(tEnd/dt)+1)

# μ = zeros(Float32,2,nT); μ[:,1] = μ0;
# Σ = zeros(Float32,2,2,nT);  Σ[:,:,1] = Σ0;

# # Linear System description
# A = 0.5f0*1.0f0I(2); # stable linear system

# #

maxval = 4.0;
xmin = maxval * [-1, -1];
xmax = maxval * [1, 1];
NN = 100;#Int(sqrt(size(C,2)));
CEval = GenerateNDGrid(xmin, xmax, [1, 1] * NN);
XX = reshape(CEval[1, :], NN, NN);
YY = reshape(CEval[2, :], NN, NN);


# for (tInt, tVal) in enumerate(tR[1:end-1])
#     μ[:,tInt+1] = A*μ[:,tInt];
#     Σ[:,:,tInt+1] = A*Σ[:,:,tInt]*A';
#     r = reshape(pdf(MvNormal(μ[:,tInt],Σ[:,:,tInt]), CEval), NN, NN);

#     figure(89); clf();
#     pcolormesh(XX, YY, r , cmap = "inferno", shading="auto"); colorbar();
#     xlabel("x1"); ylabel("x2");
#     title("ρ at t = $([tVal]);")
#     tight_layout();
#     pause(0.1);
# end

## Load data 
activFunc = tanh;
suff = string(activFunc);
nn = 48;
Q_fpke = 0.1f0#*1.0I(2); # σ^2

tEnd = 10.0f0;
dt = tEnd/10f0;

expNum = 25; 
fileLoc = "data_cont_rothe/vdp_exp$(expNum).jld2";
@info "Loading file from ts_cont_vdp exp $(expNum)"
file = jldopen(fileLoc, "r");
optParam = Array(read(file, "optParam"));
PDE_losses = read(file, "PDE_losses");
NORM_losses = read(file, "NORM_losses");
A = read(file, "A");
μ = read(file, "μ");
Σ = read(file, "Σ");
# dt_sim = 0.01;
# tR = LinRange(0.0,2, Int(0.1/dt_sim) + 1);
close(file);
println("Are any of the parameters NaN ever? $(any(isnan.(optParam)))")

mkpath("figs_cont/exp$(expNum)")

## Plot Losses
nIters = length(PDE_losses);
figure(1); clf();
semilogy(1:nIters, PDE_losses, label =  "PDE");
semilogy(1:nIters, NORM_losses, label = "NORM");
xlabel("Iterations"); ylabel("ϵ");
title("Loss Function exp$(expNum)");
title(string(" Exp $(expNum)"));
legend(); tight_layout();
savefig("figs_cont/exp$(expNum)/loss.png")

## Neural network
# chain = Chain(Dense(2, nn, activFunc), Dense(nn, 1)); # 1 layer network
chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1)); # 2 layer network
initθ,re  = Flux.destructure(chain)
phi = (x,θ) -> re(θ)(Array(x))


## controlled Dynamics do not stabilize x1
τ = 1; # for low pass filter
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
    sol = solve(prob, Tsit5(), saveat = 0.2, reltol = 1e-6, abstol = 1e-6)
    return sol
end
function nlSimGrid(x0)
    # function to simulate nonlinear controlled dynamics with initial condition x0 and controller K
    odeFn(x, p, t) = vdpDyn(x, t)
    prob = ODEProblem(odeFn, x0, (0.0, tEnd))
    sol = solve(prob, Tsit5(), saveat = dt, reltol = 1e-6, abstol = 1e-6)
    return sol
end
#
function plotTraj(sol, figNum)
    tSim = sol.t
    x1Sol = sol[1, :]
    x2Sol = sol[2, :]
    # u = [(1-exp(-τ*tSim[i]))*first(phi(sol.u[i], optParam)) for i = 1:length(tSim)]
    u = [first(phi(sol.u[i], optParam)) for i = 1:length(tSim)]

    figure(figNum);clf()
    subplot(3, 1, 1);
    PyPlot.plot(tSim, x1Sol);
    xlabel("t"); ylabel(L"x_1")
    grid("on")

    subplot(3, 1, 2)
    PyPlot.plot(tSim, x2Sol)
    xlabel("t");  ylabel(L"x_2")
    grid("on")

    subplot(3, 1, 3)
    PyPlot.plot(tSim, u)
    xlabel("t"); ylabel("u")
    tight_layout()
    grid("on")

    # suptitle("traj_$(titleSuff)");
    suptitle("Trajectory");
    tight_layout()
end

tx = -4.0 .+ 8 * rand(2);
@show tx
solSim = nlSim(tx);
println("x1 initial value: $(solSim[1,1]);  x1 terminal value: $(solSim[1,end])");
println("Terminal value state norm: $(norm(solSim[:,end]))");
figure(3);
clf();
plotTraj(solSim, 3);
savefig("figs_cont/exp$(expNum)/traj.png");

##
nEvalTerm = 10;
maxTermVal = 4.0;
xg = range(-maxTermVal, maxTermVal, length = nEvalTerm);
yg = range(-maxTermVal, maxTermVal, length = nEvalTerm);
gridXG = xg' .* ones(nEvalTerm);
gridYG = ones(nEvalTerm)' .* yg;

# plot over time
solSimGrid = [nlSimGrid([x,y]) for x in xg, y in yg];

##
μ0 = μ[:,1]; Σ0 = Σ[:,:,1];
for tInd in 1:size(solSimGrid[1,1],2)
    tVal = solSimGrid[1,1].t[tInd];

    figure(24); clf();
    # figure(23,(8,4)); clf();
    # subplot(1,2,1);
    # r = reshape(pdf(MvNormal(μ0,Σ0), CEval), NN, NN);
    # pcolormesh(XX, YY, r , cmap = "inferno", shading="auto"); colorbar();
    # title("PDF")
    # tight_layout();
    # μ0 = A*μ0;
    # Σ0 = A*Σ0*A';

    # subplot(1,2,2);
    x1TrajFull_t = [solSimGrid[i][1,tInd] for i in 1:length(solSimGrid)] 
    x2TrajFull_t = [solSimGrid[i][2,tInd] for i in 1:length(solSimGrid)];
    ax = PyPlot.axes()
    ax.set_facecolor("black");
    scatter(x1TrajFull_t, x2TrajFull_t,  c = "w", s = 10.0,);
    xlabel(L"x_1"); ylabel(L"x_2");
    xlim([-maxval, maxval]); ylim([-maxval, maxval]);
    title("t = $(tVal)")
    # title("VDP with Controller: t = $(tVal)")
    tight_layout();
    if (tVal*100%100 == 0)
    savefig("figs_cont/exp$(expNum)/scat_t$(Int(tVal)).png")
    # savefig("figs_cont/expTmp/scat_t$(Int(tVal)).png")
    end
    pause(0.001);
end


## HOW TO USE PYPLOT AXES
# f10, (ax1,ax2) = subplots(1,2, num = 10, clear = true, figsize = (8,4));
# ax1.semilogy(1:nIters, PDE_losses, label= "pde");
# ax1.legend();
# ax2.set_facecolor("black")
# ax2.scatter(1:nIters, 10.0.*PDE_losses);
# tight_layout()