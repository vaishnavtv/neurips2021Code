## Obtain a controller for vdp using PINNs
cd(@__DIR__);

using NeuralPDE, Flux, ModelingToolkit, Optimization, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra

import Random: seed!;
seed!(1);

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
# maxOptIters = 10000; # maximum number of training iterations
# opt = Optim.BFGS(); # Optimizer used for training
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

Q_fpke = 0.1f0; # Q = σ^2

# parameters for rhoSS_desired
μ_ss = zeros(Float32, 2);
Σ_ss = 0.1f0 * 1.0f0I(2);

dx = 0.05; # discretization size used for training

# file location to save data
suff = string(activFunc);
expNum = 4;
useGPU = true;
saveFile = "data/ss_cont_vdp_exp$(expNum).jld2";
runExp = true;
runExp_fileName = "out/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Controller for steady state vdp with Grid training. 2 HL with $(nn) neurons in the hl and $(suff) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS. using GPU? $(useGPU). dx = $(dx). Q_fpke = $(Q_fpke). μ_ss = $(μ_ss). Σ_ss = $(Σ_ss).
        Experiment number: $(expNum)\n")
    end
end

## set up the NeuralPDE framework using low-level API
@parameters x1, x2
@variables η(..), Kc(..)

xSym = [x1; x2]

# Van der Pol Dynamics
f(x) = [x[2]; -x[1] + (1 - x[1]^2) * x[2] + Kc(x[1], x[2])];

function g(x::Vector)
    return [0.0f0; 1.0f0]
end

# PDE
ρ(x) = exp(η(x[1], x[2]));
F = f(xSym) * ρ(xSym);
G = 0.5f0 * (g(xSym) * Q_fpke * g(xSym)') * ρ(xSym);

T1 = sum([Differential(xSym[i])(F[i]) for i = 1:length(xSym)]);
T2 = sum([
    (Differential(xSym[i]) * Differential(xSym[j]))(G[i, j]) for i = 1:length(xSym),
    j = 1:length(xSym)
]);

Eqn = expand_derivatives(-T1 + T2); # + dx*u(x1,x2)-1 ~ 0;
pdeOrig = simplify(Eqn / ρ(xSym), expand = true) ~ 0.0f0;

# derived a little bit manually to work with GPU
driftTerm = (1f0 + Differential(x2)(Kc(x1, x2)) - (x1^2)) + x2*Differential(x1)(η(x1, x2)) + (x2*(1 - (x1^2)) + Kc(x1, x2) - x1)*Differential(x2)(η(x1, x2))
diffTerm1 = Differential(x2)(Differential(x2)(η(x1, x2)))
diffTerm2 = abs2(Differential(x2)(η(x1,x2)))
diffTerm = Q_fpke/2*(diffTerm1 + diffTerm2); # diffusion term
pde = driftTerm - diffTerm ~ 0.0f0 # full pde


# Domain
maxval = 4.0f0;
domains = [x1 ∈ IntervalDomain(-maxval, maxval), x2 ∈ IntervalDomain(-maxval, maxval)];

# Boundary conditions
bcs = [ρ([-maxval, x2]) ~ 0.0f0,ρ([maxval, x2]) ~ 0f0,ρ([x1, -maxval]) ~ 0.0f0,ρ([x1, maxval]) ~ 0];

## Neural network
dim = 2 # number of dimensions
chain1 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain2 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain = [chain1, chain2];

initθ = DiffEqFlux.initial_params.(chain);
if useGPU
    using CUDA
    CUDA.allowscalar(false)
    initθ = initθ |> gpu;
end
flat_initθ = reduce(vcat, initθ);
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);


phi = NeuralPDE.get_phi.(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = [x1, x2]
depvars = [η(xSym...), Kc(xSym...)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);


_pde_loss_function = NeuralPDE.build_loss_function(pde,indvars,depvars,phi,derivative,integral,chain,initθ,strategy);

bc_indvars = NeuralPDE.get_argument(bcs, indvars, depvars);
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,phi,derivative,integral, chain,initθ,strategy,bc_indvars = bc_indvar) for (bc, bc_indvar) in zip(bcs, bc_indvars)]

train_domain_set, train_bound_set = NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);
if useGPU
    train_domain_set = train_domain_set |> gpu;
    train_bound_set = train_bound_set |> gpu;
end

pde_loss_function = NeuralPDE.get_loss_function(_pde_loss_function,train_domain_set[1],eltypeθ,parameterless_type_θ,strategy);
@show pde_loss_function(flat_initθ)

bc_loss_functions = [NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy) for (loss, set) in zip(_bc_loss_functions, train_bound_set)]

bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
@show (bc_loss_function_sum(flat_initθ))

## control loss function (requiring rhoSSPred = rhoDesired)
# rhoTrue(x) = exp(-1f0 / 2f0 * (x - μ_ss)' * inv(Σ_ss) * (x - μ_ss)) / (2 * pif0 * sqrt(det(Σ_ss))); # desired steady-state distribution (gaussian function) 
# rhoSSEq = ρ([x1, x2]) - rhoTrue([x1, x2]) ~ 0.0f0;
using Distributions
ρSS_sym = pdf(MvNormal(μ_ss, Σ_ss),xSym)
rhoSSEq = ρ([x1, x2]) - ρSS_sym ~ 0.0f0;


_rhoSS_loss_function = NeuralPDE.build_loss_function(rhoSSEq,indvars,depvars,phi,derivative,integral,chain,initθ,strategy);

rhoSS_loss_function = NeuralPDE.get_loss_function(_rhoSS_loss_function,train_domain_set[1], eltypeθ, parameterless_type_θ, strategy);
@show rhoSS_loss_function(flat_initθ);

## minimize control energy required
function uNorm_loss_function(θ)
    lenθ = length(θ)
    θ2 = θ[Int(lenθ / 2 + 1):end] # phi[2] only requires second half of parameters
    out = sum(
        (first(phi[2](train_domain_set[1][:, i], θ2)))^2 for
        i = 1:size(train_domain_set, 2)
    )
    return out
end
# @show uNorm_loss_function(flat_initθ);
##
function loss_function_(θ, p)
    return pde_loss_function(θ) + bc_loss_function_sum(θ) + rhoSS_loss_function(θ) #+    uNorm_loss_function(θ)
end

## set up Optimization optimization problem
f_ = OptimizationFunction(loss_function_, Optimization.AutoZygote())
prob = Optimization.OptimizationProblem(f_, flat_initθ)

nSteps = 0;
PDE_losses = Float32[];
BC_losses = Float32[];
rhoSS_losses = Float32[];
uNorm_losses = Float32[];
cb_ = function (p, l)
    global nSteps = nSteps + 1
    println("[$nSteps] Current loss is: $l")
    println(
        "Individual losses are: PDE loss:",
        pde_loss_function(p),
        ", BC loss:",
        bc_loss_function_sum(p),
        ", rhoSS loss:",
        rhoSS_loss_function(p),
        # ", uNorm loss:",
        # uNorm_loss_function(p),
    )

    push!(PDE_losses, pde_loss_function(p))
    push!(BC_losses, bc_loss_function_sum(p))
    push!(rhoSS_losses, rhoSS_loss_function(p))
    # push!(uNorm_losses, uNorm_loss_function(p))

    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses, BC_losses, rhoSS_losses);
    end
    return false
end

println("Calling Optimization()");
# res = Optimization.solve(prob, opt, cb = cb_, maxiters = maxOptIters);
res = Optimization.solve(prob, opt1, callback=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer)
res = Optimization.solve(prob, opt2, callback=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

## Save data
if runExp
    jldsave(saveFile; optParam = Array(res.minimizer), PDE_losses, BC_losses, rhoSS_losses)#, uNorm_losses, );
end
