## Solve the FPKE for the Van der Pol oscillator using baseline PINNs (large training set)
cd(@__DIR__);
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux

using CUDA
CUDA.allowscalar(false)

import Random:seed!; seed!(1);
using QuasiMonteCarlo

## parameters for neural network
nn = 20; # number of neurons in the hidden layer
activFunc = softplus; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 100000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

# file location to save data
nPtsPerMB = 2000;
nMB = 500;
suff = string(activFunc);
expNum = 14;
useGPU = true;
saveFile = "data_quasi/ll_quasi_vdp_exp$(expNum).jld2";
runExp = true;
runExp_fileName = "out_quasi/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Steady State vdp with QuasiMonteCarlo training. 3 HL with $(nn) neurons in the hl and $(suff) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS.  UniformSample strategy. PDE written directly in η. nPtsPerMB = $(nPtsPerMB). nMB = $(nMB). Using GPU? $(useGPU). With Softadapt, rerunning exp 13.
        Experiment number: $(expNum)\n")
    end
end
## set up the NeuralPDE framework using low-level API
@parameters x1, x2
@variables  η(..)

xSym = [x1;x2]

# Van der Pol Dynamics
f(x) = [x[2]; -x[1] + (1-x[1]^2)*x[2]];

function g(x::Vector)
    return [0.0f0;1.0f0];
end

# PDE
Q_fpke = 0.1f0; # Q = σ^2
ρ(x) = exp(η(x[1],x[2]));
F = f(xSym)*ρ(xSym);
G = 0.5f0*(g(xSym)*Q_fpke*g(xSym)')*ρ(xSym);

T1 = sum([Differential(xSym[i])(F[i]) for i in 1:length(xSym)]);
T2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);

Eqn = expand_derivatives(-T1+T2); # + dx*u(x1,x2)-1 ~ 0;
pdeOrig = simplify(Eqn/ρ(xSym)) ~ 0.0f0;
pde = pdeOrig;
# pde = (0.05f0Differential(x2)(Differential(x2)(η(x1, x2)))*exp(η(x1, x2)) + 0.05f0exp(η(x1, x2))*(Differential(x2)(η(x1, x2))^2) - (exp(η(x1, x2))*(1 - (x1^2))) - (x2*Differential(x1)(η(x1, x2))*exp(η(x1, x2))) - (Differential(x2)(η(x1, x2))*exp(η(x1, x2))*(x2*(1 - (x1^2)) - x1)))*(exp(η(x1, x2))^-1) ~ 0.0f0  # simplified pde rewritten with constants in float32 format

# Equation written directly in η
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G2 = diffC*η(xSym...);

T1_2 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(η(xSym...)))) for i in 1:length(xSym)]); # drift term
T2_2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G2[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
T2_2 += sum([(Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j]) - (Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])*η(xSym...) + diffC[i,j]*(Differential(xSym[i])(η(xSym...)))*(Differential(xSym[j])(η(xSym...))) for i in 1:length(xSym), j=1:length(xSym)]); # complete diffusion term

Eqn = expand_derivatives(-T1_2+T2_2); 
pdeOrig2 = simplify(Eqn, expand = true) ~ 0.0f0;
# pde = pdeOrig2;

# to use GPU, had to modify term: (0.05(Differential(x2)(η(x1, x2))^2))
pde = x1^2 + x1*Differential(x2)(η(x1, x2)) + 0.05f0*Differential(x2)(Differential(x2)(η(x1, x2))) + 0.05f0(abs2(Differential(x2)(η(x1, x2)))) + x2*Differential(x2)(η(x1, x2))*(x1^2) - 1.0f0 - (x2*Differential(x1)(η(x1, x2))) - (x2*Differential(x2)(η(x1, x2))) ~ 0.0f0 


## Domain
maxval = 4.0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval),
           x2 ∈ IntervalDomain(-maxval,maxval)];

# Boundary conditions
bcs = [ρ([-maxval,x2]) ~ 0.f0, ρ([maxval,x2]) ~ 0,
       ρ([x1,-maxval]) ~ 0.f0, ρ([x1,maxval]) ~ 0];

## Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));

initθ = DiffEqFlux.initial_params(chain) #|> gpu;
if useGPU
    initθ = initθ |> gpu;
end
flat_initθ = initθ
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.QuasiRandomTraining(nPtsPerMB;sampling_alg=UniformSample(), resampling=false,minibatch=nMB)

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = [x1, x2]
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


## softadapt from section 3.2.3: https://arxiv.org/pdf/2110.09813.pdf
function get_λ(nSteps)
    if nSteps == 0
        p = [1f0; 1f0];
    else
        p = softmax([PDE_losses[nSteps], BC_losses[nSteps]])
    end
    return p
end

nSteps = 0;
function loss_function_(θ, p)
    global nSteps
    λ = get_λ(nSteps)
    return λ[1]*pde_loss_function(θ) + λ[2]*bc_loss_function_sum(θ) 
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
if runExp
    jldsave(saveFile;optParam = Array(res.minimizer), PDE_losses, BC_losses);
end