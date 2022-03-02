## Solve the FPKE for vdp using nnrbf as approximating function
cd(@__DIR__);
include("../rb_nnrbf/nnrbf.jl");
include("../rb_nnrbf/libFPKE.jl");
mkpath("data_nnrbf");
mkpath("out_nnrbf");
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra

import Random:seed!; seed!(1);

using Quadrature, Cubature, Cuba

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = Optim.BFGS(); # primary optimizer used for training
maxOpt1Iters = 50; # maximum number of training iterations for opt1
# opt2 = Optim.BFGS(); # second optimizer used for fine-tuning
# maxOpt2Iters = 10; # maximum number of training iterations for opt2
Q_fpke = 0.5f0; # Q = σ^2

dx = 0.4; # discretization size used for training
nBasis = 20; # Number of basis functions in nnrbf

expNum = 2;
runExp = true;
useGPU = false;
saveFile = "data_nnrbf/vdp_exp$(expNum).jld2";
runExp_fileName = "out_nnrbf/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "NNRBF: Steady State vdp with grid training. nBasis = $(nBasis). Q_fpke = $(Q_fpke). useGPU = $(useGPU). dx = $(dx). Equation written in ρ. $(maxOpt1Iters) iterations with $(opt1).
        Experiment number: $(expNum)\n")
    end
end

## set up the NeuralPDE framework using low-level API
@parameters x1, x2
@variables  ρs(..), η(..)

xSym =  [x1;x2]

# Van der Pol Dynamics
f(x) = [x[2]; -x[1] + (1-x[1]^2)*x[2]];
g(x) = [0.0f0;1.0f0];

# PDE
println("Defining VDP PDE...");
# # PDE written in ρ
ρ(x) = ρs(x...);
F = f(xSym)*ρ(xSym);
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G = diffC*ρs(xSym...);

T1 = sum([Differential(xSym[i])(F[i]) for i in 1:length(xSym)])
T2 = sum([(Differential(xSym[i]) * Differential(xSym[j]))(G[i, j]) for j in 1:length(xSym) for i = 1:length(xSym)])
Eqn = simplify(expand_derivatives(-T1 + T2))
pde = Eqn ~ 0.0f0;
println("PDE defined.");

## Domains
maxval = 4.0f0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval),
           x2 ∈ IntervalDomain(-maxval,maxval)];

# Boundary conditions
bcs = [ρ([-maxval,x2]) ~ 0.f0, ρ([maxval,x2]) ~ 0,
       ρ([x1,-maxval]) ~ 0.f0, ρ([x1,maxval]) ~ 0];
## MODEL
d = 2; 
nParams = 2*d^2+1;
XC0 = 1 .* (-1 .+ 2*rand(d,nBasis));
XC = similar(XC0);
tf = 10.0;

XC = [propagateAdvection(f,(XC0[:,i]),tf) for i in 1:nBasis];
p2P(p,nBasis,nParams) = [p[((i-1)*nParams+1):i*nParams] for i = 1:nBasis];

ρ(P,XC,x) = sum([ϕ(p,xc,x) for (p,xc) in zip(P,XC)]);

phi = function (X,p)
    P = p2P(p, nBasis, nParams);
    return permutedims([ρ(P, XC, X[:,i]) for i in 1:size(X,2)])
end

Id = [1 0;0 1.0];
P0 = [init_params(Id,-Id,1) for i = 1:nBasis];
p0 = vcat(P0...);
initθ = p0;

chain = Chain(Dense(d,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1)); # not really being used

# initθ = DiffEqFlux.initial_params(chain);
if useGPU
    initθ = initθ |> gpu;
end
flat_initθ = initθ;
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);

# phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = [x1, x2]
depvars = [ρs(x1,x2)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

_pde_loss_function = NeuralPDE.build_loss_function(pde,indvars,depvars,phi,derivative,integral,chain,initθ,strategy,);

bc_indvars = NeuralPDE.get_argument(bcs, indvars, depvars);
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,phi,derivative,integral,chain,initθ,strategy,bc_indvars = bc_indvar) for (bc, bc_indvar) in zip(bcs, bc_indvars)]

train_domain_set, train_bound_set = NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);
if useGPU
    train_domain_set = train_domain_set |> gpu;
    train_bound_set = train_bound_set |> gpu;
end

pde_loss_function = θ -> sum(abs2, _pde_loss_function(train_domain_set[1],θ));
@show pde_loss_function(initθ)

bc_loss_functions = [θ->sum(abs2,loss(set, θ)) for (loss, set) in zip(_bc_loss_functions, train_bound_set)];

bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
@show bc_loss_function_sum(initθ)

## NORM LOSS FUNCTION
function norm_loss_function(θ)
    norm_loss = abs2(dx*d*sum(abs, phi(train_domain_set[1], θ))  - 1f0)
    return norm_loss
end
@show norm_loss_function(initθ)

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

    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $(l) \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses, BC_losses, NORM_losses, Q_fpke, XC);
    end

    return false
end

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt1, cb=cb_, maxiters=maxOpt1Iters);
# prob = remake(prob, u0=res.minimizer)
# res = GalacticOptim.solve(prob, opt2, cb=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

if runExp
    jldsave(saveFile;optParam = Array(res.minimizer), PDE_losses, BC_losses, NORM_losses, Q_fpke, XC);
end