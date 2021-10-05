## Solve the FPKE for the Van der Pol oscillator using baseline PINNs (large training set)

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux, Statistics, LinearAlgebra

using CUDA
CUDA.allowscalar(false)

import Random:seed!; seed!(1);

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = Optim.LBFGS(); # primary optimizer used for training
# opt1 = ADAM(1e-3) #Flux.Optimiser(ADAM(1e-3));
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 1000; # maximum number of training iterations for opt2

dx = [0.1f0; 0.1f0; 0.01f0]; # discretization size used for training
tEnd = 1.0f0; 
Q_fpke = 0.001f0; # Q_fpke = σ^2
α_ic = 0.0f0; # weight on initial loss

# file location to save data
suff = string(activFunc);
expNum = 4;
runExp = true;
useGPU = true;
cd(@__DIR__);
saveFile = "dataTS_grid/ll_ts_ls_exp$(expNum).jld2";
runExp_fileName = "out_grid/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Transient linear system with grid training in η. 2 HL with $(nn) neurons in the hl and $(suff) activation. $(maxOpt1Iters) iterations with LBFGS and then $(maxOpt2Iters) with LBFGS.  Q_fpke = $(Q_fpke). Using GPU. dx = $(dx). tEnd = $(tEnd). Not enforcing steady-state. Enforcing BC. Diffusion in x2.
        α_ic = $(α_ic). No IC. 
        Experiment number: $(expNum)\n")
    end
end

## set up the NeuralPDE framework using low-level API
@parameters x1, x2, t
@variables  η(..)

xSym = [x1;x2]

# Stable linear Dynamics
f(x) = -1.0f0*x; #

function g(x::Vector)
    return [0.0f0;1.0f0];
end

# PDE
Dt = Differential(t); 
Dx2 = Differential(x2);
ρ(x) = exp(η(x[1],x[2],t)); # time-varying density
# F = f(xSym)*ρ(xSym);
# G = 0.5f0*(g(xSym)*Q_fpke*g(xSym)')*ρ(xSym);

# T1 = sum([Differential(xSym[i])(F[i]) for i in 1:length(xSym)]);
# T2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
# T1 = sum([Symbolics.derivative(F[i], xSym[i]) for i = 1:length(xSym)]); # pde drift term
# T2 = sum([
#     Symbolics.derivative(Symbolics.derivative(G[i, j], xSym[i]), xSym[j]) for i = 1:length(xSym), j = 1:length(xSym)
# ]); # pde diffusion term

# pdeOrig = simplify((Dt(ρ(xSym)) + T1 - T2)/ρ(xSym), expand=true) ~ 0.0f0;
# pde = pdeOrig;

# pde = (Differential(t)(exp(η(x1, x2, t))) + Differential(x1)(x2*exp(η(x1, x2, t))) + Differential(x2)(exp(η(x1, x2, t))*(x2*(1 - (x1^2)) - x1)))*(exp(η(x1, x2, t))^-1) ~ 0.0f0 # drift term works
# diffTerm = 0.5f0*Q_fpke^2*(Differential(x2)(Differential(x2)(η(x1,x2,t))))# diffusion hessian term works
# diffTerm = ((Differential(x2)(η(x1,x2,t)))*(Differential(x2)(η(x1,x2,t)))) # square of derivative doesn't work

# # # Breaking PDE down (correct for when diffusion only in x2)
# driftTerm = T1*(exp(η(x1, x2, t))^-1)
# diffTerm1 = Differential(x2)(Differential(x2)(η(x1,x2,t))) 
# diffTerm2 = abs2(Differential(x2)(η(x1,x2,t))) # works
# diffTerm = Q_fpke/2*(diffTerm1 + diffTerm2); # diffusion term

# pde = Dt(η(x1,x2,t)) + driftTerm - diffTerm ~ 0.0f0 # full pde

##
## Writing PDE in terms of η directly - convoluted
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
ηx = η(xSym[1],xSym[2],t);
G2 = diffC*ηx;

T1 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(ηx))) for i in 1:length(xSym)]); # drift term
T2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G2[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
T2 += sum([(Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j]) - (Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])*ηx  for i in 1:length(xSym), j=1:length(xSym)]); # complete diffusion term

T2 += diffC[2,2]*abs2(Differential(x2)(ηx)); # only when diffusion in x2
# T2 = sum([diffC[i,j]*((Differential(xSym[i])(ηx))*(Differential(xSym[j])(ηx))) for i in 1:length(xSym), j=1:length(xSym)]) # works only when 



Eqn = Dt(ηx) + expand_derivatives(-T1+T2); 
pdeOrig2 = simplify(Eqn, expand = true) ~ 0.0f0;

pde = pdeOrig2;

## Domain
maxval = 4.0; 
domains = [x1 ∈ IntervalDomain(-maxval,maxval),
           x2 ∈ IntervalDomain(-maxval,maxval),
           t ∈ IntervalDomain(0.0f0, tEnd)];

ssExp  =  Dt((η(x1,x2,tEnd)));
ρ_ic(x) = exp(η(x[1],x[2],0.0f0)); 
μ_ss = [0.0f0,0.0f0]; 
Σ_ss = 0.1f0*[1.0f0; 1.0f0] .* 1.0f0I(2);
inv_Σ_ss = Float32.(inv(Σ_ss))
sqrt_det_Σ_ss = Float32(sqrt(det(Σ_ss)));
icExp = ρ_ic(xSym)

ρ0(x) =  exp(-0.5f0 * (x - μ_ss)' * inv_Σ_ss * (x - μ_ss)) / (2.0f0 * Float32(pi) * sqrt_det_Σ_ss); # ρ at t0, Gaussian

# Initial and Boundary conditions
bcs = [ρ([-maxval,x2]) ~ 0.0f0, ρ([maxval,x2]) ~ 0.0f0,
       ρ([x1,-maxval]) ~ 0.0f0, ρ([x1,maxval]) ~ 0.0f0];# 
    #    icExp ~ ρ0([x1,x2])];#, # initial condition
    #    ssExp ~ 0.0f0]; # steady-state condition

## Neural network
dim = 3 # number of dimensions
chain = Chain(Dense(dim,nn, activFunc), Dense(nn,nn,activFunc), Dense(nn,1));

initθ = DiffEqFlux.initial_params(chain) #|> gpu;
if useGPU
    initθ = initθ |> gpu;
end
flat_initθ = initθ
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);


phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = [x1, x2, t]
depvars = [η(x1,x2,t)]

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

# ic_loss_function = (θ) -> sum(abs2,_bc_loss_functions[5](train_bound_set[5], θ));
# @show ic_loss_function(initθ)

bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
@show bc_loss_function_sum(initθ)

function loss_function_(θ, p)
    # return pde_loss_function(θ) + α_ic*ic_loss_function(θ)
    return pde_loss_function(θ) + bc_loss_function_sum(θ) #+ α_ic*ic_loss_function(θ)
end
@show loss_function_(initθ,0)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, initθ)

nSteps = 0;
PDE_losses = Float32[];
BC_losses = Float32[];
# IC_losses = Float32[];
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
        bc_loss_function_sum(p)#,
        # ", IC loss:",
        # ic_loss_function(p)
    )

    push!(PDE_losses, pde_loss_function(p))
    push!(BC_losses, bc_loss_function_sum(p))
    # push!(IC_losses, ic_loss_function(p))

    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses, BC_losses);#, IC_losses);
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
    jldsave(saveFile;optParam = Array(res.minimizer), PDE_losses, BC_losses);#, IC_losses);
end