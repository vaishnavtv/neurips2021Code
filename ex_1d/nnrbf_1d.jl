## Solve the FPKE for the 1d example using nnrbf as approximating function
cd(@__DIR__);
include("../rb_nnrbf/nnrbf.jl");
include("../rb_nnrbf/libFPKE.jl");
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra

import Random:seed!; seed!(1);

using Quadrature, Cubature, Cuba

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2
Q_fpke = 0.25f0; # Q = σ^2

dx = 0.01; # discretization size used for training
nBasis = 50; # Number of basis functions in nnrbf

expNum = 3;
runExp = true;
useGPU = false;
saveFile = "data_nnrbf/eta_exp$(expNum).jld2";
runExp_fileName = "out_nnrbf/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "NNRBF: Steady State 1D with grid training. nBasis = $(nBasis). Q_fpke = $(Q_fpke). useGPU = $(useGPU). dx = $(dx). Equation written in ρ.
        $(maxOpt1Iters) iterations with $(opt1) and then $(maxOpt2Iters) with $(opt2).
        Experiment number: $(expNum)\n")
    end
end

## set up the NeuralPDE framework using low-level API
@parameters x1
@variables  ρs(..), η(..)

xSym = x1;

# 1D Dynamics
α = 0.3f0; β = 0.5f0;
f(x) = α*x - β*x^3
g(x) = 1.0f0;

# PDE
ρ_true(x) = exp((1/(2*Q_fpke))*(2*α*x^2 - β*x^4)); # true analytical solution, before normalziation

println("Defining PDE...");
# # PDE written in ρ
ρ(x) = ρs(x...);
F = f(xSym)*ρ(xSym);
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G = diffC*ρs(xSym...);

T1 = sum([Differential(xSym[i])(F[i]) for i in 1:length(xSym)])
T2 = sum([(Differential(xSym[i]) * Differential(xSym[j]))(G[i, j]) for j in 1:length(xSym) for i = 1:length(xSym)])
Eqn = simplify(expand_derivatives(-T1 + T2))
pde = Eqn ~ 0.0f0;

# #  PDE written directly in η
# ρ(x) = exp(η(x...));
# F = f(xSym)*ρ(xSym);
# diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
# G = diffC*η(xSym...);

# T1 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(η(xSym...)))) for i in 1:length(xSym)]); # drift term
# T2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
# T2 += sum([(Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])  - (Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])*η(xSym...) + diffC[i,j]*abs2(Differential(xSym[i])(η(xSym...))) for i in 1:length(xSym), j=1:length(xSym)]); # complete diffusion term, modified for GPU

# Eqn = expand_derivatives(-T1+T2); 
# pdeOrig = simplify(Eqn, expand = true) ~ 0.0f0;
# pde = pdeOrig;
# #
println("PDE defined.");

## Domain
maxval = 2.2f0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval)];

# Boundary conditions
bcs = [ρ(-maxval) ~ 0.0f0, ρ(maxval) ~ 0.0f0]

## Neural network
dim = 1 # number of dimensions
Id = collect(Float32.(I(dim)));
xc = Float32.(0.1f0*randn(nBasis));
chain = Chain(Parallel(vcat,[NNRBF([1.0f0;;],[xc[i]],[-1.0f0;;],[xc[i]]) for i = 1:nBasis]), Linear(ones(Float32,(1,nBasis))));
# chain = Chain(Parallel(vcat,[begin xc = 0.1f0*randn(dim); NNRBF(Id,xc,-Id,xc) end for i = 1:nBasis]), Linear(ones(Float32,1,nBasis))) # incorrect number of parameters
# chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));
# chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));

initθ = DiffEqFlux.initial_params(chain);
if useGPU
    initθ = initθ |> gpu;
end
flat_initθ = initθ;
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = [x1]
depvars = [ρs(x1)]

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
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);
if useGPU
    train_domain_set = train_domain_set |> gpu;
    train_bound_set = train_bound_set |> gpu;
end

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

## NORM LOSS FUNCTION
function norm_loss_function(θ)
    norm_loss = sum(abs2, [sum(phi(x, θ)) for x in train_domain_set[1]])  - 1f0
    return norm_loss
end
@show norm_loss_function(initθ)
##


nSteps = 0;
function loss_function_(θ, p)
    return pde_loss_function(θ) + bc_loss_function_sum(θ) + norm_loss_function(θ)
end
@show loss_function_(initθ,0)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, initθ)

nSteps = 0;
PDE_losses = Float32[];
BC_losses = Float32[];
NORM_losses = Float32[];
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
        ", NORM loss:",
        norm_loss_function(p)
    )

    push!(PDE_losses, pde_loss_function(p))
    push!(BC_losses, bc_loss_function_sum(p))
    push!(NORM_losses, norm_loss_function(p))

    return false
end

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt1, cb=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer)
res = GalacticOptim.solve(prob, opt2, cb=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

if runExp
    jldsave(saveFile;optParam = Array(res.minimizer), PDE_losses, BC_losses);
end