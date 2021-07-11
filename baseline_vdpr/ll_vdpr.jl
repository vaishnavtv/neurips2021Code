## Solve the FPKE for the Van der Pol Rayleigh oscillator using baseline PINNs (large training set)

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, Symbolics, JLD2
import Random:seed!; seed!(1);

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
maxOptIters = 10000; # maximum number of training iterations
opt = Optim.BFGS(); # Optimizer used for training


dx = 0.05

suff = string(activFunc);
saveFile = "data/dx5eM2_vdpr_$(suff)_$(nn).jld2";

# Van der Pol Rayleigh Dynamics
@parameters x1, x2
@variables  η(..)

x = [x1;x2]

# Dynamics
f(x) = [x[2]; -x[1] + (1-x[1]^2 - x[2]^2)*x[2]];

function g(x)
    return [0.0;1.0];
end

# PDE
Q = 0.3; # Q = σ^2

ρ(x) = exp(η(x[1],x[2]));
F = f(x)*ρ(x);
G = 0.5*(g(x)*Q*g(x)')*ρ(x);

T1 = sum([Differential(x[i])(F[i]) for i in 1:length(x)]);
T2 = sum([(Differential(x[i])*Differential(x[j]))(G[i,j]) for i in 1:length(x), j=1:length(x)]);

Eqn = expand_derivatives(-T1+T2); # + dx*u(x1,x2)-1 ~ 0;
pde = simplify(Eqn/ρ(x),expand=true) ~ 0;

# Domain
maxval = 2.0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval),
           x2 ∈ IntervalDomain(-maxval,maxval)];

# Boundary conditions
bcs = [ρ([-maxval,x2]) ~ 0.f0, ρ([maxval,x2]) ~ 0,
       ρ([x1,-maxval]) ~ 0.f0, ρ([x1,maxval]) ~ 0];

## Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));

strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain);
derivative = NeuralPDE.get_numeric_derivative();
initθ = DiffEqFlux.initial_params(chain)

indvars = [x1, x2]
depvars = [η]

_pde_loss_function = NeuralPDE.build_loss_function(pde, indvars, depvars, phi, derivative, initθ, strategy);

bc_indvars = NeuralPDE.get_argument(bcs, indvars, depvars);
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars, phi, derivative,initθ,strategy, bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]

pde_train_set,bcs_train_set = NeuralPDE.generate_training_sets(domains,dx,[pde],bcs,indvars,depvars);

pde_bounds, bcs_bounds = NeuralPDE.get_bounds(domains,[pde], bcs,indvars,depvars);

pde_loss_function = NeuralPDE.get_loss_function([_pde_loss_function], pde_train_set, strategy)

bc_loss_function = NeuralPDE.get_loss_function(_bc_loss_functions, bcs_train_set, strategy)

function loss_function_(θ,p)
    return pde_loss_function(θ) + bc_loss_function(θ)
end

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, initθ)  

nSteps = 0;
PDE_losses = Float32[]; BC_losses = Float32[]; 
cb_ = function (p,l)
    global nSteps = nSteps + 1;
    println("[$nSteps] Current loss is: $l")
    println("Individual losses are: PDE loss:", pde_loss_function(p), ", BC loss:",  bc_loss_function(p))

    push!(PDE_losses, pde_loss_function(p))
    push!(BC_losses, bc_loss_function(p))
    return false
end

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt, cb = cb_, maxiters=maxOptIters);
println("Optimization done.");

## Save data
jldsave(saveFile;optParam = res.minimizer, PDE_losses, BC_losses);