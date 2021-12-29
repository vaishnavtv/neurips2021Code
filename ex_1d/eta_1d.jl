## Solve the FPKE for the 1d example using baseline PINNs (large training set)
cd(@__DIR__);
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra

import Random:seed!; seed!(1);

using Quadrature, Cubature, Cuba

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = Optim.LBFGS(); # primary optimizer used for training
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

dx = 0.01; # discretization size used for training

## set up the NeuralPDE framework using low-level API
@parameters x1
@variables  η(..)

xSym = x1;

# 1D Dynamics
α = 0.3; β = 0.5;
f(x) = α*x - β*x^3

g(x) = 1.0f0;

# PDE
Q_fpke = 0.25f0; # Q = σ^2
ρ(x) = exp(η(xSym));
F = f(xSym)*ρ(xSym);
G = 0.5f0*(g(xSym)*Q_fpke*g(xSym)')*ρ(xSym);

T1 = sum([Differential(xSym[i])(F[i]) for i in 1:length(xSym)]);
T2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);

Eqn = expand_derivatives(-T1+T2); # + dx*u(x1,x2)-1 ~ 0;
pdeOrig = simplify(Eqn/ρ(xSym)) ~ 0.0f0;
pde = pdeOrig;

## Domain
maxval = 2.2f0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval)];

# Boundary conditions
bcs = [ρ([-maxval]) ~ 0.f0, ρ([maxval]) ~ 0]

## Neural network
dim = 1 # number of dimensions
chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));
# chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));

initθ = DiffEqFlux.initial_params(chain)
flat_initθ = initθ;
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);


phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = [x1]
depvars = [η(x1)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

_pde_loss_function = NeuralPDE.build_loss_function(
    pde,
    indvars,
    depvars,
    phi,
    derivative,
    integral,
    chain,
    initθ,
    strategy,
);

bc_indvars = NeuralPDE.get_argument(bcs, indvars, depvars);
_bc_loss_functions = [
    NeuralPDE.build_loss_function(
        bc,
        indvars,
        depvars,
        phi,
        derivative,
        integral,
        chain,
        initθ,
        strategy,
        bc_indvars = bc_indvar,
    ) for (bc, bc_indvar) in zip(bcs, bc_indvars)
]

train_domain_set, train_bound_set =
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars) ;

pde_loss_function = NeuralPDE.get_loss_function(
    _pde_loss_function,
    train_domain_set[1],
    eltypeθ,
    parameterless_type_θ,
    strategy,
);
@show pde_loss_function(initθ)

bc_loss_functions = [
    NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy) for
    (loss, set) in zip(_bc_loss_functions, train_bound_set)
]

bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
@show bc_loss_function_sum(initθ)

nSteps = 0;
function loss_function_(θ, p)
    return pde_loss_function(θ) + bc_loss_function_sum(θ) 
end
@show loss_function_(initθ,0)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, initθ)

nSteps = 0;
PDE_losses = Float32[];
BC_losses = Float32[];
cb_ = function (p, l)
    if any(isnan.(p))
        println("SOME PARAMETERS ARE NaN.")
    end

    global nSteps = nSteps + 1
    println("[$nSteps] Current loss is: $l")
    println(
        "Individual losses are: PDE loss:",
        pde_loss_function(p),
        ", BC loss:",
        bc_loss_function_sum(p),
    )

    push!(PDE_losses, pde_loss_function(p))
    push!(BC_losses, bc_loss_function_sum(p))

    return false
end

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt1, cb=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer)
res = GalacticOptim.solve(prob, opt2, cb=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

## Plot results
import ModelingToolkit: Interval, infimum, supremum
using PyPlot; pygui(true);
xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains][1]
u_predict  = [exp(first(phi([x],res.minimizer))) for x in xs]

figure(1); clf();
plot(xs ,u_predict, label = "predict");
xlabel("x"); ylabel("ρ");
title("Steady-state Solution");
tight_layout(); 
# savefig("figs_eta/hl2nn48Tanh.png");

figure(2); clf();
nIters = length(PDE_losses);
semilogy(1:nIters, PDE_losses, label = "PDE");
semilogy(1:nIters, BC_losses, label = "BC");
title("training loss");
xlabel("Iteration"); ylabel("ϵ")
tight_layout();