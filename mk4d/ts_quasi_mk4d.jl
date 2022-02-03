# Dynamic system 5 from Kumar's PUFEM paper using quasi-strategy
# system taken from Wojtkiewicz, S. F., and L. A. Bergman. "Numerical solution of high dimensional Fokker-Planck equations." 8th ASCE Specialty Conference on Probablistic Mechanics and Structural Reliability, Notre Dame, IN, USA. 2000.
cd(@__DIR__);
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux

import Random:seed!; seed!(1);
using QuasiMonteCarlo

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 100000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

# file location to save data
tEnd = 0.1f0;
nPtsPerMB = 5000;
nMB = 100;
suff = string(activFunc);
expNum = 1;
useGPU = false;
saveFile = "data_ts_quasi/ll_quasi_mk4d_exp$(expNum).jld2";
runExp = true;
runExp_fileName = "out_ts_quasi/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Transient State 4D linear dynamics with QuasiMonteCarlo training. 2 HL with $(nn) neurons in the hl and $(suff) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS.  UniformSample strategy. PDE written directly in η. nPtsPerMB = $(nPtsPerMB). nMB = $(nMB). Using GPU? $(useGPU). PDE fixed to use GPU.
        Experiment number: $(expNum)\n")
    end
end
## set up the NeuralPDE framework using low-level API
@parameters t, x1, x2, x3, x4
@variables  η(..)

xSym = [x1;x2; x3; x4]

# Linear 4D Dynamics
k1 = 1; k2 = 1; k3 = 1;
c1 = 0.4; c2 = 0.4;
linA = Float32.([0 1 0 0;
        -(k1+k2) -c1 k2 0;
        0 0 0 1;
        k2 0 -(k2+k3) -c2]);
f(x) = linA*x;

function g(x::Vector)
    return [0.0f0 0.0f0;1.0f0 0.0f0; 0.0f0 0.0f0; 0.0f0 1.0f0];
end

# PDE
println("Defining PDE");
Q_fpke = 0.4f0; # Q = σ^2
ρ(x) = exp(η(t, x...));

# Equation written directly in η
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G2 = diffC*η(t, xSym...);

dtT = Differential(t)(η(t, xSym...));
T1_2 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(η(t, xSym...)))) for i in 1:length(xSym)]); # drift term
# T2_2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G2[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
# T2_2 += sum([(Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j]) - (Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])*η(xSym...) + diffC[i,j]*(Differential(xSym[i])(η(xSym...)))*(Differential(xSym[j])(η(xSym...))) for i in 1:length(xSym), j=1:length(xSym)]); # complete diffusion term
T2_2 = 0.2f0*abs2(Differential(x2)(η(t, x1, x2, x3, x4))) + 0.2f0*abs2(Differential(x4)(η(t, x1, x2, x3, x4))) + 0.2f0Differential(x2)(Differential(x2)(η(t, x1, x2, x3, x4))) + 0.2f0Differential(x4)(Differential(x4)(η(t, x1, x2, x3, x4))); # diffusion term modified to avoid NaNs when using ADAM

Eqn = expand_derivatives(dtT + T1_2 - T2_2); 
pdeOrig2 = simplify(Eqn, expand = true) ~ 0.0f0;
pde = pdeOrig2;
println("PDE in η defined symbolically.")

## Domain
maxval = 4.0f0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval),
           x2 ∈ IntervalDomain(-maxval,maxval),
           x3 ∈ IntervalDomain(-maxval,maxval),
           x4 ∈ IntervalDomain(-maxval,maxval),
           t ∈ IntervalDomain(0.0f0, tEnd)];

# Boundary conditions
bcs = [ρ([-maxval,x2,x3,x4]) ~ 0.f0, ρ([maxval,x2,x3,x4]) ~ 0,
       ρ([x1,-maxval,x3,x4]) ~ 0.f0, ρ([x1,maxval,x3,x4]) ~ 0,
       ρ([x1,x2,-maxval,x4]) ~ 0.f0, ρ([x1,x2,maxval,x4]) ~ 0,
       ρ([x1,x2,x3,-maxval]) ~ 0.f0, ρ([x1,x2,x3,maxval]) ~ 0,];

## Neural network
dim = 5 # number of dimensions
chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));

initθ = DiffEqFlux.initial_params(chain) #|> gpu;
if useGPU
    using CUDA
    CUDA.allowscalar(false)
    initθ = initθ |> gpu;
end
flat_initθ = initθ
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.QuasiRandomTraining(nPtsPerMB;sampling_alg=UniformSample(), resampling=false,minibatch=nMB)

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = [t; xSym];
depvars = [η(t, xSym...)]

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

bc_indvars = NeuralPDE.get_variables(bcs, indvars, depvars);
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

pde_bounds, bcs_bounds = NeuralPDE.get_bounds(domains, [pde], bcs, eltypeθ, indvars, depvars, strategy)

pde_loss_function = NeuralPDE.get_loss_function(
    _pde_loss_function,
    pde_bounds[1],
    eltypeθ,
    parameterless_type_θ,
    strategy,
);
@show pde_loss_function(initθ)

bc_loss_functions = [
    NeuralPDE.get_loss_function(loss, bound, eltypeθ, parameterless_type_θ, strategy) for
    (loss, bound) in zip(_bc_loss_functions, bcs_bounds)
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
    jldsave(saveFile;optParam = Array(res.minimizer), PDE_losses, BC_losses);
end