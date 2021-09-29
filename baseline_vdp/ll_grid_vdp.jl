## Solve the FPKE for the Van der Pol oscillator using baseline PINNs (large training set)
cd(@__DIR__);
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux

# using CUDA
# CUDA.allowscalar(false)
import Random:seed!; seed!(1);

using Quadrature, Cubature, Cuba

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = Optim.BFGS(); # primary optimizer used for training
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 1000; # maximum number of training iterations for opt2

dx = 0.05; # discretization size used for training

# file location to save data
suff = string(activFunc);
expNum = 8;
saveFile = "data_grid/ll_grid_vdp_exp$(expNum).jld2";
useGPU = false;
runExp = true;
runExp_fileName = "out_grid/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Steady State vdp with Grid training. 2 HL with $(nn) neurons in the hl and $(suff) activation. $(maxOpt1Iters) iterations with BFGS and then $(maxOpt2Iters) with LBFGS. Not using GPU. 
        Adding norm loss. Using CubaDivonne.
        Experiment number: $(expNum)\n")
    end
end
## set up the NeuralPDE framework using low-level API
@parameters x1, x2
@variables  η(..)

x = [x1;x2]

# Van der Pol Dynamics
f(x) = [x[2]; -x[1] + (1-x[1]^2)*x[2]];

function g(x::Vector)
    return [0.0f0;1.0f0];
end

# PDE
Q_fpke = 0.1f0; # Q = σ^2
ρ(x) = exp(η(x[1],x[2]));
F = f(x)*ρ(x);
G = 0.5f0*(g(x)*Q_fpke*g(x)')*ρ(x);

T1 = sum([Differential(x[i])(F[i]) for i in 1:length(x)]);
T2 = sum([(Differential(x[i])*Differential(x[j]))(G[i,j]) for i in 1:length(x), j=1:length(x)]);

Eqn = expand_derivatives(-T1+T2); # + dx*u(x1,x2)-1 ~ 0;
pdeOrig = simplify(Eqn/ρ(x)) ~ 0.0f0;
# pde = pdeOrig;
# pde = (0.05f0Differential(x2)(Differential(x2)(η(x1, x2)))*exp(η(x1, x2)) + 0.05f0exp(η(x1, x2))*(Differential(x2)(η(x1, x2))^2) - (exp(η(x1, x2))*(1 - (x1^2))) - (x2*Differential(x1)(η(x1, x2))*exp(η(x1, x2))) - (Differential(x2)(η(x1, x2))*exp(η(x1, x2))*(x2*(1 - (x1^2)) - x1)))*(exp(η(x1, x2))^-1) ~ 0.0f0  # simplified pde rewritten with constants in float32 format

# pde = (Differential(x1)(x2*exp(η(x1, x2))) + Differential(x2)(exp(η(x1, x2))*(x2*(1 - (x1^2)) - x1)))*(exp(η(x1, x2))^-1) ~ 0.0f0 # drift term (works, no NaN)
# pde = Differential(x2)(Differential(x2)((η(x1, x2)))) ~ 0.0f0  # diffusion term 1 (works, no NaN)
# pde = ((Differential(x2)(η(x1,x2,t)))*(Differential(x2)(η(x1,x2,t)))) ~ 0.0f0 # square of derivative doesn't work

driftTerm = (Differential(x1)(x2*exp(η(x1, x2))) + Differential(x2)(exp(η(x1, x2))*(x2*(1 - (x1^2)) - x1)))*(exp(η(x1, x2))^-1)
diffTerm1 = Differential(x2)(Differential(x2)(η(x1,x2))) 
diffTerm2 = abs2(Differential(x2)(η(x1,x2))) # works
diffTerm = Q_fpke/2*(diffTerm1 + diffTerm2); # diffusion term

pde = driftTerm - diffTerm ~ 0.0f0 # full pde

## Domain
maxval = 4.0f0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval),
           x2 ∈ IntervalDomain(-maxval,maxval)];

# Boundary conditions
bcs = [ρ([-maxval,x2]) ~ 0.f0, ρ([maxval,x2]) ~ 0,
       ρ([x1,-maxval]) ~ 0.f0, ρ([x1,maxval]) ~ 0];

## Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));

initθ = DiffEqFlux.initial_params(chain) 
if useGPU
    initθ = initθ |> gpu;
end
flat_initθ = initθ
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);


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
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars) ;
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

## additional loss function
lbs = [-maxval, -maxval]; ubs = [maxval, maxval];
function norm_loss_function(θ)
    function inner_f(x,θ)
         return exp(sum(phi(x, θ))) # density
    end
    prob = QuadratureProblem(inner_f, lbs, ubs, θ)
    norm2 = solve(prob, CubaDivonne(), reltol = 1e-3, abstol = 1e-3);
    return abs2(norm2[1] - 1)
end
@show norm_loss_function(initθ)

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
    )

    push!(PDE_losses, pde_loss_function(p))
    push!(BC_losses, bc_loss_function_sum(p))
    push!(NORM_losses, norm_loss_function(p))

    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses, BC_losses, NORM_losses );
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
    jldsave(saveFile;optParam = Array(res.minimizer), PDE_losses, BC_losses, NORM_losses);
end