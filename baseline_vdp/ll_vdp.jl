## Solve the FPKE for the Van der Pol oscillator using baseline PINNs (large training set)

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux
import Random:seed!; seed!(1);

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
maxOptIters = 10000; # maximum number of training iterations
opt = Optim.BFGS(); # Optimizer used for training


dx = 0.05; # discretization size used for training

# file location to save data
suff = string(activFunc);
saveFile = "data/dx5eM2_vdp_$(suff)_$(nn)_2.jld2";

## set up the NeuralPDE framework using low-level API
@parameters x1, x2
@variables  η(..)

x = [x1;x2]

# Van der Pol Dynamics
f(x) = [x[2]; -x[1] + (1-x[1]^2)*x[2]];

function g(x::Vector)
    return [0.0;1.0];
end

# PDE
Q = 0.1; # Q = σ^2
ρ(x) = exp(η(x[1],x[2]));
F = f(x)*ρ(x);
G = 0.5*(g(x)*Q*g(x)')*ρ(x);

T1 = sum([Differential(x[i])(F[i]) for i in 1:length(x)]);
T2 = sum([(Differential(x[i])*Differential(x[j]))(G[i,j]) for i in 1:length(x), j=1:length(x)]);

Eqn = expand_derivatives(-T1+T2); # + dx*u(x1,x2)-1 ~ 0;
pde = simplify(Eqn/ρ(x),expand=true) ~ 0.0f0;

# Domain
maxval = 4.0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval),
           x2 ∈ IntervalDomain(-maxval,maxval)];

# Boundary conditions
bcs = [ρ([-maxval,x2]) ~ 0.f0, ρ([maxval,x2]) ~ 0,
       ρ([x1,-maxval]) ~ 0.f0, ρ([x1,maxval]) ~ 0];

## Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));

initθ = DiffEqFlux.initial_params(chain)
flat_initθ = if (typeof(chain) <: AbstractVector)
    reduce(vcat, initθ)
else
    initθ
end
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);


phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = [x1, x2]
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

# bc_indvars = NeuralPDE.get_argument(bcs, indvars, depvars);
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
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars) ;#|> gpu;

pde_loss_function = NeuralPDE.get_loss_function(
    _pde_loss_function,
    train_domain_set[1],
    eltypeθ,
    parameterless_type_θ,
    strategy,
);
@show pde_loss_function(initθ)
# sleep(1000)
bc_loss_functions = [
    NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy) for
    (loss, set) in zip(_bc_loss_functions, train_bound_set)
]

bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
typeof(bc_loss_function_sum(initθ))
function loss_function_(θ, p)
    return pde_loss_function(θ) + bc_loss_function_sum(θ)  #loss_function__(θ)
    # return pde_loss_function(θ)
    # return bc_loss_function_sum(θ)
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
    return false
end

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt, cb = cb_, maxiters = maxOptIters);
println("Optimization done.");

## Save data
jldsave(saveFile;optParam = res.minimizer, PDE_losses, BC_losses);
