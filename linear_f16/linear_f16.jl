## Solve the FPKE for the Van der Pol Rayleigh oscillator using baseline PINNs (large training set)
cd(@__DIR__);
include("f16_controller.jl")

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, Symbolics, JLD2
using F16Model
using CUDA
CUDA.allowscalar(false)

import Random: seed!;
seed!(1);

# parameters for neural network
nn = 100; # number of neurons in the hidden layers
activFunc = tanh; # activation function
maxOptIters = 100000; # maximum number of training iterations
opt = Optim.LBFGS(); # Optimizer used for training
# opt = ADAM(1e-3); 

suff = string(activFunc);
dx = [10.0;1.0;1.0;0.01]; # grid discretization in V (ft/s), α (deg), θ (deg), q (rad/s)

saveFile = "data/linear_f16_t1.jld2";

# Nominal Controller for Longitudinal F16Model trimmmed at specified altitude and velocity 
# Trim vehicle at specified altitude and velocity
h0 = 10000; # ft
Vt0 = 500;   # ft/s
# xbar, ubar, status, prob = F16Model.Trim(h0, Vt0); # Default is steady-level
A2, B2 = getLinearModel4x(xbar, ubar);
# Kc = getKc(A2, B2)
function f16_linDyn(x)
    # linear 4 state perturbation dynamics
    u = Kc * x
    dx = A2 * x + B2 * u
    return (dx)
end    

@parameters xV, xα, xθ, xq
@variables η(..)

x = [xV, xα, xθ, xq]

f(x) = f16_linDyn(x)

function g(x)
    return [1.0; 0.0; 0.0; 0.0] # noise only in first state for now
end

# PDE
Q = 0.3; # Q = σ^2

ρ(x) = exp(η(x[1], x[2], x[3], x[4]));
F = f(x) * ρ(x);
G = 0.5 * (g(x) * Q * g(x)') * ρ(x);

T1 = sum([Differential(x[i])(F[i]) for i = 1:length(x)]);
T2 = sum([
    (Differential(x[i]) * Differential(x[j]))(G[i, j]) for i = 1:length(x), j = 1:length(x)
]);
Eqn = expand_derivatives(-T1 + T2); # + dx*u(x1,x2)-1 ~ 0;
pdeOrig = simplify(Eqn / ρ(x)) ~ 0.0f0; 
pde = pdeOrig;

## Domain
xV_min = 100;
xV_max = 1500;
xα_min = deg2rad(-10);
xα_max = pi / 4;
xθ_min = xα_min;
xθ_max = xα_max;
xq_min = 0;
xq_max = 1.0;
domains = [
    xV ∈ IntervalDomain(xV_min, xV_max),
    xα ∈ IntervalDomain(xα_min, xα_max),
    xθ ∈ IntervalDomain(xθ_min, xθ_max),
    xq ∈ IntervalDomain(xq_min, xq_max),
];

## Boundary conditions
bcs = [
    ρ([xV_min, xα, xθ, xq]) ~ 0.0f0,
    ρ([xV_max, xα, xθ, xq]) ~ 0.0f0,
    ρ([xV, xα_min, xθ, xq]) ~ 0.0f0,
    ρ([xV, xα_max, xθ, xq]) ~ 0.0f0,
    ρ([xV, xα, xθ_min, xq]) ~ 0.0f0,
    ρ([xV, xα, xθ_max, xq]) ~ 0.0f0,
    ρ([xV, xα, xθ, xq_min]) ~ 0.0f0,
    ρ([xV, xα, xθ, xq_max]) ~ 0.0f0,
];

## Neural network
dim = 4 # number of dimensions
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));#|> gpu;

initθ = DiffEqFlux.initial_params(chain)|> gpu;
flat_initθ = initθ;
eltypeθ = eltype(flat_initθ);

parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = x
depvars = [η]

_pde_loss_function = NeuralPDE.build_loss_function(
    pde,
    indvars,
    depvars,
    phi,
    derivative,
    chain,
    initθ,
    strategy,
);

bc_indvars = NeuralPDE.get_variables(bcs, indvars, depvars);
_bc_loss_functions = [
    NeuralPDE.build_loss_function(
        bc,
        indvars,
        depvars,
        phi,
        derivative,
        chain,
        initθ,
        strategy,
        bc_indvars = bc_indvar,
    ) for (bc, bc_indvar) in zip(bcs, bc_indvars)
]

train_domain_set, train_bound_set =
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);# |> gpu;
train_domain_set = train_domain_set |> gpu

##
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

function loss_function_(θ, p)
    return pde_loss_function(θ) + bc_loss_function_sum(θ)
end
@show loss_function_(initθ, 0)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, initθ)

nSteps = 0;
PDE_losses = Float32[];
BC_losses = Float32[];
cb_ = function (p, l)
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
res = GalacticOptim.solve(prob, opt, cb = cb_, maxiters = maxOptIters);
println("Optimization done.");

## Save data
cd(@__DIR__);
jldsave(saveFile; optParam = Array(res.minimizer), PDE_losses, BC_losses);
