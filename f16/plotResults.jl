## Plot the results of the simulation for baseline linear F16
using JLD2,
    PyPlot,
    NeuralPDE,
    ModelingToolkit,
    LinearAlgebra,
    Flux,
    Trapz,
    Printf,
    LaTeXStrings,
    ForwardDiff,
    Quadrature
pygui(true);
# @variables x1, x2

# Load data 
activFunc = tanh;
dx = 0.05;
suff = string(activFunc);
nn = 100;
optFlag = 1;

Q = 0.3;

cd(@__DIR__);
fileLoc = "data/baseline_f16_ADAM_gpu_4_30.jld2";

println("Loading file");
file = jldopen(fileLoc, "r");
optParam = read(file, "optParam");
PDE_losses = read(file, "PDE_losses");
term1_losses = read(file, "term1_losses");
term2_losses = read(file, "term2_losses");
BC_losses = read(file, "BC_losses");
close(file);


## plot losses
figure(1); clf();
nIters = length(PDE_losses)
semilogy(1:nIters, PDE_losses, label= "PDE_losses");
semilogy(1:nIters, term1_losses, label= "term1_losses");
semilogy(1:nIters, term2_losses, label= "term2_losses");
semilogy(1:nIters, BC_losses, label = "BC_losses");
legend();
tight_layout();
# savefig("figs/ADAM_100k.png");
## Neural network
dim = 4;
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));#|> gpu;
parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

## System Dynamics
include("f16_controller.jl")
A2, B2 = getLinearModel4x(xbar, ubar);
# Kc = getKc(A2, B2)
function f16_linDyn(x)
    # linear 4 state perturbation dynamics
    u = Kc * x
    dx = A2 * x + B2 * u
    return (dx)
end    
f(x) = f16_linDyn(x)


## Generate functions to check solution and PDE error
function ρ_pdeErr_fns(optParam)
    function ηNetS(x)
        return first(phi(x, optParam))
    end
    ρNetS(x) = exp(ηNetS(x)) # solution after first iteration
    # df(x) = ForwardDiff.jacobian(f, x)
    # dη(x) = ForwardDiff.gradient(ηNetS, x)
    # d2η(x) = ForwardDiff.jacobian(dη, x)

    # pdeErrFn(x) = tr(df(x)) + dot(f(x), dη(x)) - Q / 2 * (d2η(x)[end] + (dη(x)[end])^2)
    return ρNetS#, pdeErrFn
end
ρFn = ρ_pdeErr_fns(optParam);
ρFn(xbar[ind_x])
exp(phi(xbar[ind_x], optParam)[1]) # should be same as line above
## Domains
xV_min = 100;
xV_max = 1500;
xα_min = deg2rad(-10);
xα_max = pi / 4;
xθ_min = xα_min;
xθ_max = xα_max;
xq_min = 0;
xq_max = 1.0;

nEvalFine = 10; # to be changed

xVFine = range(xV_min, xV_max, length = nEvalFine);
xαFine = range(xα_min, xα_max, length = nEvalFine);
xθFine = range(xθ_min, xθ_max, length = nEvalFine);
xqFine = range(xq_min, xq_max, length = nEvalFine);
##
RHOFine = [ρFn([xvg, xαg, xθg, xqg]) for xvg in xVFine, xαg in xαFine, xθg in xθFine, xqg in xqFine];

ρFnQuad(x,p) = ρFn(x);
prob = QuadratureProblem(ρFnQuad, [xV_min,xα_min,xθ_min, xq_min], [xV_max, xα_max, xθ_max, xq_max], p = 0);
sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)

# normalize
RHOFine = RHOFine/sol.u;
# FFFine = [pdeErrFn([x, y])^2 for x in xxFine, y in yyFine];

## Quadrature technique to obtain marginal pdf, requires integration for every set of states
# Supposing we want the marginal pdf for the first two states(V, α)
function ρ12(xV,xα)
    # p = [xV, xα]; y = [xθ, xq]
    ρFnQuad34(y,p) = ρFn([xV;xα;y])
    prob = QuadratureProblem(ρFnQuad34, [xθ_min; xq_min], [xθ_max; xq_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end
vBar = xbar[ind_x][1]; αBar = xbar[ind_x][2];
ρ12(vBar,αBar)

RHO12Fine = [ρ12(xvg, xαg) for xvg in xVFine, xαg in xαFine];
# println("The mean squared equation error with dx=$(dx) is:")
# mseEqErr = sum(FFFine[:] .^ 2) / length(FFFine);
# @show mseEqErr;
# mseEqErrStr = @sprintf "%.2e" mseEqErr;
# XXFine = similar(RHOFine);
# YYFine = similar(RHOFine);

# for i = 1:nEvalFine, j = 1:nEvalFine
#     XXFine[i, j] = xxFine[i]
#     YYFine[i, j] = yyFine[j]
# end

## Plot shown in paper
function plotDistErr(figNum)

    figure(figNum, [8, 4])
    clf()
    subplot(1, 2, 1)
    # figure(figNum); clf();
    pcolor(XXFine, YYFine, RHOFine, shading = "auto", cmap = "inferno")
    colorbar()
    xlabel("x1")
    ylabel("x2")
    axis("auto")
    title("Steady-State Solution (ρ)")

    subplot(1, 2, 2)
    pcolormesh(XXFine, YYFine, FFFine, cmap = "inferno", shading = "auto")
    colorbar()
    axis("auto")
    title(L"Equation Error; $ϵ_{pde}$ = %$(mseEqErrStr)")
    xlabel("x1")
    ylabel("x2")

    tight_layout()

end
plotDistErr(3);

