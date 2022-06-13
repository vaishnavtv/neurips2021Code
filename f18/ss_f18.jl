## Obtain a controller for f18 using PINNs
# With diffusion, NAN issue
cd(@__DIR__);
include("f18Dyn.jl")
mkpath("out_ss")
mkpath("data_ss")

using NeuralPDE, Flux, ModelingToolkit, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra, Distributions, Optimization, GPUArrays

import Random: seed!;
seed!(1);

## parameters for neural network
nn = 100; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 50000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

Q_fpke = 0.1f0; # Q = σ^2

# file location to save data
expNum = 7;
useGPU = true;
runExp = true;
saveFile = "data_ss/exp$(expNum).jld2";
runExp_fileName = "out_ss/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Finding the ss distribution for the trimmed F18. 3 HL with $(nn) neurons in the hl and $(activFunc) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS. using GPU? $(useGPU). Q_fpke = $(Q_fpke). Increased size of training dataset. Changed dx. Added diffusion terms. Q_fpke is still 0.
        Experiment number: $(expNum)\n")
    end
end

## set up the NeuralPDE framework using low-level API
@parameters x1, x2, x3, x4
@variables η(..)

xSym = [x1; x2; x3; x4]

##
maskIndx = zeros(Float32,(length(f18_xTrim),length(indX)));
maskIndu = zeros(Float32,(length(f18_uTrim),length(indU)));
for i in 1:length(indX)
    maskIndx[indX[i],i] = 1f0;
    if i<=length(indU)
        maskIndu[indU[i],i] = 1f0;
    end
end

# F18 Dynamics
function f(xd)
    # xd: perturbed state

    # maskTrim = ones(Float32,length(f18_xTrim)); maskTrim[indX] .= 0f0;
    xFull = f18_xTrim + maskIndx*xd; # perturbed state
    uFull = f18_uTrim; # utrim

    xdotFull = f18Dyn(xFull, uFull)

    return (xdotFull[indX]) # return the 4 state dynamics

end

##
g(x::Vector) = [1.0f0; 1.0f0;1.0f0; 1.0f0] # diffusion vector needs to be modified

# PDE
ρ(x) = exp(η(x...));
F = f(xSym) * ρ(xSym);
G = 0.5f0 * (g(xSym) * Q_fpke * g(xSym)') * ρ(xSym);

# Drift Terms
driftTerms = ([Differential(xSym[i])(F[i]) for i = 1:length(xSym)]);
pdeDrift = driftTerms./(ρ(xSym)) .~ 0.0f0

# Diffusion
pdeDiff = Equation[]; # need to do the following to avoid NaNs
# (Differential(x1)(η(x1,x2,x3,x4)))^2 gives NaNs
# abs2(Differential(x1)(η(x1,x2,x3,x4))) is NaN-safe

for i in 1:4
    for j in 1:4
        if i!=j
            offDiagTerm = (Differential(xSym[i])*Differential(xSym[j]))(G[i,j])
            offDiagTerm = expand_derivatives(offDiagTerm)
            eqn = simplify(offDiagTerm/ρ(xSym), expand = true) ~ 0.0f0
            push!(pdeDiff, eqn);
        else
            eqn = 0.5f0*Q_fpke*g(xSym)[i]*(abs2(Differential(xSym[i])(η(xSym...))) + Differential(xSym[i])(Differential(xSym[i])(η(xSym...)))) ~ 0.0f0
            push!(pdeDiff, eqn);
        end
    end
end
println("PDE defined.")

## Domain
x1_min = -100f0; x1_max = 100f0;
x2_min = deg2rad(-10f0); x2_max = deg2rad(10f0);
x3_min = x2_min ; x3_max = x2_max ;
x4_min = deg2rad(-5f0); x4_max = deg2rad(5f0);

domains = [x1 ∈ IntervalDomain(x1_min, x1_max), x2 ∈ IntervalDomain(x2_min, x2_max), x3 ∈ IntervalDomain(x3_min, x3_max), x4 ∈ IntervalDomain(x4_min, x4_max),];

dx = [10f0; deg2rad(1f0); deg2rad(1f0); deg2rad(1f0);]; # discretization size used for training

# Boundary conditions
bcs = [η(-100f0,x2,x3,x4) ~ 0.f0, η(100f0,x2,x3,x4) ~ 0.f0,
       η(x1,x2_min,x3,x4) ~ 0.f0, η(x1,x2_max,x3,x4) ~ 0.f0,
       η(x1,x2,x3_min,x4) ~ 0.f0, η(x1,x2,x3_max,x4) ~ 0.f0,
       η(x1,x2,x3,x4_min) ~ 0.f0, η(x1,x2_max,x3,x4_max) ~ 0.f0,];


## Neural network set up
dim = 4 # number of dimensions
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));

initθ = DiffEqFlux.initial_params(chain);
if useGPU
    using CUDA
    CUDA.allowscalar(false)
    initθ = initθ |> gpu;
end 
flat_initθ = initθ; 
th0 = flat_initθ;
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);

indvars = xSym
depvars = [η(xSym...)]

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();
integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

tx = cu(f18_xTrim[indX]);

## Loss function
println("Defining pde loss function for drift and diffusion terms.")
_pde_loss_functions_drift = [NeuralPDE.build_loss_function(pde_i, indvars, depvars, phi, derivative, integral, chain, initθ, strategy) for pde_i in pdeDrift];
@show [fn(tx, th0) for fn in _pde_loss_functions_drift]

_pde_loss_functions_diff = [NeuralPDE.build_loss_function(pde_i, indvars, depvars, phi, derivative, integral, chain, initθ, strategy) for pde_i in pdeDiff];
@show [fn(tx, th0) for fn in _pde_loss_functions_diff]


_pde_loss_function(cord, θ) =  sum([fn(cord, θ) for fn in _pde_loss_functions_drift]) .- sum([fn(cord, θ) for fn in _pde_loss_functions_diff])
@show _pde_loss_function(tx, th0) 
# println("sleeping here.")
# sleep(10000);

bc_indvars = NeuralPDE.get_argument(bcs, indvars, depvars);
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,phi,derivative,integral,chain,initθ,strategy,bc_indvars = bc_indvar) for (bc, bc_indvar) in zip(bcs, bc_indvars)]

# Domain and training sets
train_domain_set, train_bound_set =
    NeuralPDE.generate_training_sets(domains, dx, pdeDrift, bcs, eltypeθ, indvars, depvars);
if useGPU
    train_domain_set = train_domain_set |> gpu;
    train_bound_set = train_bound_set |> gpu;
end

using Statistics
pde_loss_function = (θ) -> mean(abs2,_pde_loss_function(cu(train_domain_set[1]), θ));
@show pde_loss_function(flat_initθ)

bc_loss_functions = [NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy) for (loss, set) in zip(_bc_loss_functions, train_bound_set)]

bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
@show bc_loss_function_sum(initθ)


loss_function_(θ, p) =  pde_loss_function(θ) + bc_loss_function_sum(θ)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, Optimization.AutoZygote())
prob = Optimization.OptimizationProblem(f_, flat_initθ)

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

    push!(PDE_losses, l)
    push!(BC_losses, bc_loss_function_sum(p))

    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses);
    end
    return false
end

println("Calling GalacticOptim()");
res = Optimization.solve(prob, opt1, callback=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer);
res = Optimization.solve(prob, opt2, callback=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

## Save data
if runExp
    jldsave(saveFile;optParam = Array(res.minimizer),PDE_losses, BC_losses);
end