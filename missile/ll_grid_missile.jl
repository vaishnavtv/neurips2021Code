## Solve the FPKE for the missile using baseline PINNs (large training set)

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, Symbolics, JLD2
cd(@__DIR__);
include("cpu_missileDynamics.jl");
# using CUDA
# CUDA.allowscalar(false)

import Random: seed!;
seed!(1);

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = Optim.BFGS(); # primary optimizer used for training
maxOpt1Iters = 10000 # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

α_bc = 1.0;

## Grid discretization
dM = 0.01; dα = 0.01;
dx = [dM; dα] # grid discretization in M, α (rad)

Q_fpke = 0.01f0;#*1.0I(2); # σ^2


suff = string(activFunc);
expNum = 5;
useGPU = false;
runExp = true; 
saveFile = "data_grid/ll_grid_missile_exp$(expNum).jld2";
runExp_fileName = "out_grid/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Missile with GridTraining and dx = $(dx). 2 HL with $(nn) neurons in the hl and $(suff) activation. Boundary loss coefficient: $(α_bc). $(maxOpt1Iters) iterations with BFGS and $(maxOpt2Iters) with LBFGS. 
        Diffusion coefficient Q_fpke = $(Q_fpke). In both states.
        Forward Time Dynamics.
        dx = $(dx). Not using GPU.
        Experiment number: $(expNum)\n")
    end
end
# Van der Pol Rayleigh Dynamics
@parameters x1, x2
@variables η(..)

xSym = [x1; x2]

# PDE
ρ(x) = exp(η(x...));
F = f(xSym) * ρ(xSym); # drift term
diffC = 0.5f0 * (g(xSym) * Q_fpke * g(xSym)'); # diffusion coefficient
G = diffC * ρ(xSym); # diffusion term

T1 = sum([Symbolics.derivative(F[i], xSym[i]) for i = 1:length(xSym)]); # pde drift term
T2 = sum([
    Symbolics.derivative(Symbolics.derivative(G[i, j], xSym[i]), xSym[j]) for i = 1:length(xSym), j = 1:length(xSym)
]); # pde diffusion term


Eqn = expand_derivatives(-T1 + T2); # + dx*u(x1,x2)-1 ~ 0;
pdeOrig = simplify(Eqn / ρ(xSym)) ~ 0.0f0;
pde = pdeOrig;

## Domain
minM = 1.2; maxM = 2.5;
minα = -1.0; maxα = 1.5;

domains = [x1 ∈ IntervalDomain(minM, maxM), x2 ∈ IntervalDomain(minα, maxα)];

# Boundary conditions
bcs = [
    ρ([minM, x2]) ~ 0.0f0,
    ρ([maxM, x2]) ~ 0.0f0,
    ρ([x1, minα]) ~ 0.0f0,
    ρ([x1, maxα]) ~ 0.0f0,
];

## Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1)) ;
# chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1)) ;

initθ = DiffEqFlux.initial_params(chain) #|> gpu;
if useGPU
    initθ = initθ |> gpu;
end
flat_initθ = initθ
eltypeθ = eltype(flat_initθ)

parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = xSym
depvars = [η(x1,x2)]

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
# @show _pde_loss_function(xTrim_Ma, initθ)
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
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Size of training dataset: $(size(train_domain_set[1],2))\n")
    end
end;

pde_loss_function = NeuralPDE.get_loss_function(
    _pde_loss_function,
    train_domain_set[1],
    eltypeθ,
    parameterless_type_θ,
    strategy,
);
println("Checking if loss functions are computed for initial parameters:");
@show pde_loss_function(initθ)

bc_loss_functions = [
    NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy) for
    (loss, set) in zip(_bc_loss_functions, train_bound_set)
]
bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
typeof(bc_loss_function_sum(initθ))
function loss_function_(θ, p)
    return pde_loss_function(θ) + α_bc*bc_loss_function_sum(θ)  
end
@show bc_loss_function_sum(initθ)
@show loss_function_(initθ,0)

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

    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses, BC_losses);
    end
    return false
end

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt1, cb=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer)
res = GalacticOptim.solve(prob, opt2, cb=cb_, maxiters=maxOpt2Iters);

println("Optimization done.");

## Save data
cd(@__DIR__);
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Size of training dataset: $(size(train_domain_set[1],2))\n")
    end;
    jldsave(saveFile;optParam = Array(res.minimizer), PDE_losses, BC_losses);
end