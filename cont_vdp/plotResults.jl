## Plot the results of the bsaeline-PINNs implementation for the Van der Pol Rayleigh oscillator
using JLD2,
    PyPlot,
    NeuralPDE,
    ModelingToolkit,
    LinearAlgebra,
    Flux,
    Trapz,
    Printf,
    DiffEqFlux,
    ForwardDiff
@variables x1, x2
# gr();
# pygui(true);
import Random;
Random.seed!(1);

# Load data 
activFunc = tanh;
suff = string(activFunc);
nn = 48;
Q = 0.1;

# parameters for rhoSS_desired
μ_ss = zeros(2);
Σ_ss = 0.001 * 1.0I(2);
rhoTrue(x) = exp(-1 / 2 * (x - μ_ss)' * inv(Σ_ss) * (x - μ_ss)) / (2 * pi * sqrt(det(Σ_ss))); # desired steady-state distribution (gaussian function) 

cd(@__DIR__);
fileLoc = "data/dx5eM2_vdp_$(suff)_$(nn)_contMin.jld2";

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
clf;
nLosses = length(PDE_losses);
semilogy(1:nLosses, PDE_losses, label = "pde");
semilogy(1:nLosses, BC_losses, label = "bc");
semilogy(1:nLosses, rhoSS_losses, label = "rhoSS");
# semilogy(1:nLosses, uNorm_losses, label = "uNorm");
xlabel("Iterations");
ylabel("ϵ");
legend();
##

parameterless_type_θ = DiffEqBase.parameterless_type(optParam);

chain1 = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain2 = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain = [chain1; chain2];

phi = NeuralPDE.get_phi.(chain, parameterless_type_θ);

maxval = 2.0f0;

nEvalFine = 100;
len1 = Int(length(optParam) / 2);
optParam1 = optParam[1:len1];
optParam2 = optParam[len1+1:end];

xs = range(-maxval, maxval, length = nEvalFine);
ys = range(-maxval, maxval, length = nEvalFine)
rho_pred = Float32.([exp(first(phi[1]([x, y], optParam1))) for x in xs, y in ys]);
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
# using Plots; gr();
pygui(true);
figure(1, (12, 4));
clf();
subplot(1, 3, 1);
p1 = pcolor(gridX, gridY, rho_pred_norm, cmap = "inferno", shading = "auto");
colorbar();
xlabel("x1");
ylabel("x2");
title("Prediction");
axis("auto");
tight_layout();

subplot(1, 3, 2);
PyPlot.pcolor(gridX, gridY, rho_true_norm, cmap = "inferno", shading = "auto");
colorbar();
xlabel("x1");
ylabel("x2");
title("Exact");
axis("auto");
tight_layout();

errNorm = abs.(rho_true_norm - rho_pred_norm);
mseRHOErr = sum(errNorm[:] .^ 2) / length(errNorm);
mseRHOErrStr = @sprintf "%.2e" mseRHOErr;

subplot(1, 3, 3);
PyPlot.pcolor(gridX, gridY, errNorm, cmap = "inferno", shading = "auto");
colorbar();
xlabel("x1");
ylabel("x2");
title(L"Solution Error; $ϵ_{ρ}=$ %$mseRHOErrStr");
axis("auto");
tight_layout();


## controlled Dynamics do not stabilize x1
using DifferentialEquations
# Dynamics with controller Kc
function vdpDyn(x)
    dx = similar(x)
    # u = first(phi[2](x, optParam));
    u = first(phi[2](x, optParam2))
    dx[1] = x[2]
    dx[2] = -x[1] + (1 - x[1]^2) * x[2] + u
    return (dx)
end
tx = zeros(2)
#
tEnd = 500.0;
function nlSim(x0)
    # function to simulate nonlinear controlled dynamics with initial condition x0 and controller K
    odeFn(x, p, t) = vdpDyn(x)
    prob = ODEProblem(odeFn, x0, (0.0, tEnd))
    sol = solve(prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)
    return sol
end
#
function plotTraj(sol, figNum)
    tSim = sol.t
    x1Sol = sol[1, :]
    x2Sol = sol[2, :]
    u = [first(phi[2](sol.u[i], optParam)) for i = 1:length(tSim)]

    figure(figNum)
    clf()
    subplot(3, 1, 1)
    PyPlot.plot(tSim, x1Sol)
    xlabel("t")
    ylabel("x1")
    grid("on")
    subplot(3, 1, 2)
    PyPlot.plot(tSim, x2Sol)
    xlabel("t")
    ylabel("x2")
    grid("on")
    subplot(3, 1, 3)
    PyPlot.plot(tSim, u)
    xlabel("t")
    ylabel("u")
    tight_layout()
    grid("on")

    # suptitle("traj_$(titleSuff)");
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

## check terminal values over grid
nEvalTerm = 10;
maxTermVal = 5;
xg = range(-maxTermVal, maxTermVal, length = nEvalTerm);
yg = range(-maxTermVal, maxTermVal, length = nEvalTerm);
gridXG = xg' .* ones(nEvalTerm);
gridYG = ones(nEvalTerm)' .* yg;
termNorm = [norm((nlSim([x, y])[:, end])) for x in xg, y in yg];
##
figure(90);
clf();
pcolor(gridXG, gridYG, termNorm, shading = "auto");
xlabel("x1");
ylabel("x2");
title("Norm of terminal states after 500 seconds for different x0");
colorbar();
##
function f(x)
    u_ = first(phi[2](x, optParam2))
    out = [x[2]; -x[1] + (1 - x[1]^2) * x[2] + u_]
    return out
end
# f(x) = [x[2]; -x[1] + (1-x[1]^2)*x[2] + first(phi[2]([x[1], x[2]], optParam2))];

function ρ_pdeErr_fns(optParam)
    function ηNetS(x)
        return first(phi[1](x, optParam))
    end
    ρNetS(x) = exp(ηNetS(x)) # solution after first iteration
    df(x) = ForwardDiff.jacobian(z -> f(z), x)
    dη(x) = ForwardDiff.gradient(ηNetS, x)
    d2η(x) = ForwardDiff.jacobian(dη, x)


    pdeErrFn(x) = tr(df(x)) + dot(f(x), dη(x)) - Q / 2 * (d2η(x)[end] + (dη(x)[end])^2)

    return ρNetS, pdeErrFn

end
ρFn, pdeErrFn = ρ_pdeErr_fns(optParam1);
pdeErrFine = [pdeErrFn([x, y])^2 for x in xs, y in ys];
mseErrFine = sum(pdeErrFine[:] .^ 2) / length(pdeErrFine);
@show mseErrFine
##
using ForwardDiff
p(x) = x + deepcopy(x);
ForwardDiff.derivative(p, 3)

ϵ_pde = mseErrFine;
@show ϵ_pde
