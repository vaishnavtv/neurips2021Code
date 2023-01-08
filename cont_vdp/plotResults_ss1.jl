## Plot the results of cont_ss1 the Van der Pol  oscillator
cd(@__DIR__)
using JLD2,PyPlot,LinearAlgebra,Flux,Trapz,Printf, LaTeXStrings
pygui(true);
include("../utils.jl")
mkpath("figs_ss1")

maxval = 4.0;
xmin = maxval * [-1, -1];
xmax = maxval * [1, 1];
NN = 100;#Int(sqrt(size(C,2)));
CEval = GenerateNDGrid(xmin, xmax, [1, 1] * NN);
XX = reshape(CEval[1, :], NN, NN);
YY = reshape(CEval[2, :], NN, NN);

# Load data 
activFunc = tanh;
suff = string(activFunc);
nn = 48;
Q_fpke = 0.1;
tEnd = 20.0f0;
dt = tEnd/10f0;

# parameters for rhoSS_desired
μ_ss = zeros(2);
Σ_ss = 0.1 * 1.0I(2);
rhoTrue(x) = exp(-1 / 2 * (x - μ_ss)' * inv(Σ_ss) * (x - μ_ss)) / (2 * pi * sqrt(det(Σ_ss))); # desired steady-state distribution (gaussian function) 

expNum = 9;
# fileLoc = "data/dx5eM2_vdp_tanh_48_cont.jld2";
fileLoc = "data/ss_cont_vdp_exp$(expNum).jld2";
@info "Loading file from cont_vdp_ss1 exp $(expNum)"

mkpath("figs_ss1/exp$(expNum)")

## Load data
println("Loading file");
file = jldopen(fileLoc, "r");
optParam = read(file, "optParam");
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
rhoSS_losses = read(file, "rhoSS_losses");
# uNorm_losses = read(file, "uNorm_losses");
close(file);

## Plot losses over time
figure(569);
clf();
nLosses = length(PDE_losses);
semilogy(1:nLosses, PDE_losses, label = "PDE");
semilogy(1:nLosses, BC_losses, label = "BC");
semilogy(1:nLosses, rhoSS_losses, label = "RHO_SS");
# semilogy(1:nLosses, uNorm_losses, label = "uNorm");
xlabel("Iterations");
ylabel("ϵ");
title("Loss Functions");
legend();
tight_layout();
savefig("figs_ss1/exp$(expNum)/loss.png");
##

chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
# chain2 = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
# chain = [chain1; chain2];

initθ,re  = Flux.destructure(chain)
phi = (x,θ) -> re(θ)(Array(x))

maxval = 4.0f0;

nEvalFine = 100;
len1 = Int(length(optParam) / 2);
optParam1 = optParam[1:len1];
optParam2 = optParam[len1+1:end];

xs = range(-maxval, maxval, length = nEvalFine);
ys = range(-maxval, maxval, length = nEvalFine)
rho_pred = Float32.([exp(first(phi([x, y], optParam1))) for x in xs, y in ys]);
rho_true = Float32.([rhoTrue([x, y]) for x in xs, y in ys]);
##
rho_pred_norm = rho_pred / trapz((xs, ys), rho_pred);
rho_true_norm = rho_true / trapz((xs, ys), rho_true);

gridX = xs' .* ones(nEvalFine);
gridY = ones(nEvalFine)' .* ys;

rho_pred_norm = Float32.(reshape(rho_pred_norm, (length(ys), length(xs))));
rho_true_norm = Float32.(reshape(rho_true_norm, (length(ys), length(xs))));
gridX = Float32.(gridX);
gridY = Float32.(gridY);
##
figure(1, (12, 4));
clf();
subplot(1, 3, 1);
p1 = pcolor(gridX, gridY, rho_pred_norm, cmap = "inferno", shading = "auto");
colorbar();
xlabel(L"x_1");
ylabel(L"x_2");
title("Prediction");
axis("auto");
tight_layout();

subplot(1, 3, 2);
PyPlot.pcolor(gridX, gridY, rho_true_norm, cmap = "inferno", shading = "auto");
colorbar();
xlabel(L"x_1");
ylabel(L"x_2");
title("Exact");
axis("auto");
tight_layout();

errNorm = abs.(rho_true_norm - rho_pred_norm);
mseRHOErr = sum(errNorm[:] .^ 2) / length(errNorm);
mseRHOErrStr = @sprintf "%.2e" mseRHOErr;

subplot(1, 3, 3);
PyPlot.pcolor(gridX, gridY, errNorm, cmap = "inferno", shading = "auto");
colorbar();
xlabel(L"x_1");
ylabel(L"x_2");
title(L"Solution Error; $ϵ_{mse}=$ %$mseRHOErrStr");
axis("auto");
tight_layout();

savefig("figs_ss1/exp$(expNum)/soln.png");

## controlled Dynamics do not stabilize x1
using DifferentialEquations
# Dynamics with controller Kc
function vdpDyn(x)
    dx = similar(x)
    u = first(phi(x, optParam2))
    dx[1] = x[2]
    dx[2] = -x[1] + (1 - x[1]^2) * x[2] + u
    return (dx)
end
tx = zeros(2)
#
function nlSim(x0)
    # function to simulate nonlinear controlled dynamics with initial condition x0 and controller K
    odeFn(x, p, t) = vdpDyn(x)
    prob = ODEProblem(odeFn, x0, (0.0, tEnd))
    sol = solve(prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)
    return sol
end

function nlSimGrid(x0)
    # function to simulate nonlinear controlled dynamics with initial condition x0 and controller K
    # higher dt 
    odeFn(x, p, t) = vdpDyn(x)
    prob = ODEProblem(odeFn, x0, (0.0, tEnd))
    sol = solve(prob, Tsit5(), saveat = dt,  reltol = 1e-6, abstol = 1e-6)
    return sol
end
#
function plotTraj(sol, figNum)
    tSim = sol.t
    x1Sol = sol[1, :]
    x2Sol = sol[2, :]
    u = [first(phi(sol.u[i], optParam2)) for i = 1:length(tSim)]

    figure(figNum); clf();
    subplot(3, 1, 1)
    PyPlot.plot(tSim, x1Sol)
    xlabel("t"); ylabel(L"x_1")
    grid("on")

    subplot(3, 1, 2)
    PyPlot.plot(tSim, x2Sol)
    xlabel("t"); ylabel(L"x_2")
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
plotTraj(solSim, 3)
savefig("figs_ss1/exp$(expNum)/traj.png");


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
for tInd in 1:size(solSimGrid[1,1],2)
    tVal = solSimGrid[1,1].t[tInd];

    figure(24); clf();
    x1TrajFull_t = [solSimGrid[i][1,tInd] for i in 1:length(solSimGrid)] 
    x2TrajFull_t = [solSimGrid[i][2,tInd] for i in 1:length(solSimGrid)];
    ax = PyPlot.axes()
    ax.set_facecolor("black");
    scatter(x1TrajFull_t, x2TrajFull_t,  c = "w", s = 10.0,);
    xlabel(L"x_1"); ylabel(L"x_2");
    xlim([-maxval, maxval]); ylim([-maxval, maxval]);
    title("t = $(tVal)")
    # title("Steady-state ρ Control: Q = $(Q_fpke), t = $(tVal)")
    tight_layout();
    if (tVal*100%100 == 0)
        savefig("figs_ss1/exp$(expNum)/scat_t$(Int(tVal)).png")
    end
    pause(0.001);
end