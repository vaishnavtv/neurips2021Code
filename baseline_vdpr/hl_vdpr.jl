## Solve the FPKE for the Van der Pol Rayleigh oscillator using baseline PINNs (large training set)

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, Symbolics, JLD2
using CUDA

import Random: seed!;
seed!(1);

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
maxOptIters = 50000; # maximum number of training iterations
opt = Optim.LBFGS(); # Optimizer used for training
# opt = ADAM(1e-3);
CUDA.allowscalar(false)

dx = 0.05

suff = string(activFunc);
saveFile = "data/dx5eM2_vdpr_$(suff)_$(nn)_gpu_hl_LB.jld2";

# Van der Pol Rayleigh Dynamics
@parameters x1, x2
@variables η(..)

x = [x1; x2]

# Dynamics
f(x) = [x[2]; -x[1] + (1 - x[1]^2 - x[2]^2) * x[2]];

function g(x)
    return [0.0f0; 1.0f0]
end

# PDE
Q = 0.3f0; # Q = σ^2

ρ(x) = exp(η(x[1], x[2]));
F = f(x) * ρ(x);
G = 0.5f0 * (g(x) * Q * g(x)') * ρ(x);

T1 = sum([Differential(x[i])(F[i]) for i = 1:length(x)]);
T2 = sum([
    (Differential(x[i]) * Differential(x[j]))(G[i, j]) for i = 1:length(x), j = 1:length(x)
]);

Eqn = expand_derivatives(-T1 + T2); # + dx*u(x1,x2)-1 ~ 0;
pdeOrig = simplify(Eqn / ρ(x)) ~ 0.0f0;
pde = (0.15f0Differential(x2)(Differential(x2)(η(x1, x2)))*exp(η(x1, x2)) + 0.15f0exp(η(x1, x2))*(Differential(x2)(η(x1, x2))^2) - (exp(η(x1, x2))*(1 - (x1^2) - (3(x2^2)))) - (x2*Differential(x1)(η(x1, x2))*exp(η(x1, x2))) - (Differential(x2)(η(x1, x2))*exp(η(x1, x2))*(x2*(1 - (x1^2) - (x2^2)) - x1)))*(exp(η(x1, x2))^-1) ~ 0.0f0; # simplified pde rewritten with constants in float32 format

# Domain
maxval = 2.0;
domains = [x1 ∈ IntervalDomain(-maxval, maxval), x2 ∈ IntervalDomain(-maxval, maxval)];

# Boundary conditions
bcs = [
    ρ([-maxval, x2]) ~ 0.0f0,
    ρ([maxval, x2]) ~ 0.0f0,
    ρ([x1, -maxval]) ~ 0.0f0,
    ρ([x1, maxval]) ~ 0.0f0,
];

## Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1)) |> gpu;

initθ = DiffEqFlux.initial_params(chain) |> gpu;

strategy = NeuralPDE.GridTraining(dx);

discretization = NeuralPDE.PhysicsInformedNN(chain, strategy, init_params = initθ);

indvars = [x1, x2]
depvars = [η]

cb_ = function (p,l)
    println("Current loss is: $l")
    return false
end

pde_system = PDESystem(pde, bcs, domains, indvars, depvars);
prob = NeuralPDE.discretize(pde_system, discretization);
res = GalacticOptim.solve(prob, opt, cb = cb_, maxiters = maxOptIters);
phi = discretization.phi;


## Save data
cd(@__DIR__)
jldsave(saveFile;optParam = Array(res.minimizer));#, PDE_losses, BC_losses);
