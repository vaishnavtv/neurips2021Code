## Obtain a controller for vdp using PINNs
# unlike cont_ll_vdp.jl, terminal state is fixed.
cd(@__DIR__);
mkpath("out_rhoConst")
mkpath("data_rhoConst")

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra, Distributions

import Random: seed!;
seed!(1);

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

# parameters for rhoSS_desired
μ_ss = [0f0,0f0];
Σ_ss = 1f0*1.0f0I(2);

dx = 0.05f0; # discretization size used for training
Q_fpke = 0.0f0; # Q = σ^2

# file location to save data
expNum = 11;
useGPU = true;
runExp = true;
saveFile = "data_rhoConst/exp$(expNum).jld2";
runExp_fileName = "out_rhoConst/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Generating a controller for vdp with desired ss distribution. 2 HL with $(nn) neurons in the hl and $(activFunc) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS. using GPU? $(useGPU). dx = $(dx). Q_fpke = $(Q_fpke). μ_ss = $(μ_ss). Σ_ss = $(Σ_ss). Not dividing equation by ρ.
        Experiment number: $(expNum)\n")
    end
end

## set up the NeuralPDE framework using low-level API
@parameters x1, x2
@variables Kc(..)

xSym = [x1; x2]

# Van der Pol Dynamics
f(x) = [x[2]; -x[1] + (1f0 - x[1]^2) * x[2] + Kc(x[1], x[2])];

g(x::Vector) = [0.0f0; 1.0f0]

# PDE
ρSS_sym = pdf(MvNormal(μ_ss, Σ_ss),xSym);
F = f(xSym) * ρSS_sym;
G = 0.5f0 * (g(xSym) * Q_fpke * g(xSym)') * ρSS_sym;

T1 = sum([Differential(xSym[i])(F[i]) for i = 1:length(xSym)]);
T2 = sum([(Differential(xSym[i]) * Differential(xSym[j]))(G[i, j]) for i = 1:length(xSym),j = 1:length(xSym)]);

Eqn = expand_derivatives(-T1 + T2); # + dx*u(x1,x2)-1 ~ 0;
pde = simplify(Eqn, expand = true) ~ 0.0f0;

println("PDE defined.")
# sleep(1000)

# Domain
maxval = 4.0f0;
domains = [x1 ∈ IntervalDomain(-maxval, maxval), x2 ∈ IntervalDomain(-maxval, maxval)];

# Boundary conditions
bcs = [Kc(-maxval,x2) ~ 0.f0, Kc(maxval,x2) ~ 0];


## Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));

initθ = DiffEqFlux.initial_params(chain);
if useGPU
    using CUDA
    CUDA.allowscalar(false)
    initθ = initθ |> gpu;
end
eltypeθ = eltype(initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(initθ);

strategy = NeuralPDE.GridTraining(dx);

indvars = [x1, x2]
depvars = [Kc(xSym...)]

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();
integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);


_pde_loss_function = NeuralPDE.build_loss_function(pde, indvars, depvars, phi, derivative, integral, chain, initθ, strategy);

bc_indvars = NeuralPDE.get_argument(bcs, indvars, depvars);
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,phi,derivative,integral, chain, initθ, strategy, bc_indvars = bc_indvar) for (bc, bc_indvar) in zip(bcs, bc_indvars)]

train_domain_set, train_bound_set =
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);
if useGPU
    train_domain_set = train_domain_set |> gpu;
    train_bound_set = train_bound_set |> gpu;
end

using Statistics
pde_loss_function = (θ) -> mean(abs2,_pde_loss_function(train_domain_set[1], θ));
@show pde_loss_function(initθ)

loss_function_(θ, p) =  pde_loss_function(θ)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, initθ)

nSteps = 0;
PDE_losses = Float32[];
cb_ = function (p, l)
    if any(isnan.(p))
        println("SOME PARAMETERS ARE NaN.")
    end

    global nSteps = nSteps + 1
    println("[$nSteps] Current loss is: $l")

    push!(PDE_losses, l)

    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses);
    end
    return false
end

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt1, callback=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer);
res = GalacticOptim.solve(prob, opt2, callback=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

## Save data
if runExp
    jldsave(saveFile;optParam = Array(res.minimizer),PDE_losses);
end