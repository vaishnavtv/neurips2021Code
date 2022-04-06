## Solve the transient FPKE for the Van der Pol oscillator using baseline PINNs (large training set) and Rothe's method
# Grid strategy
cd(@__DIR__);
mkpath("out_rothe");
mkpath("data_rothe");
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra, Distributions

import Random:seed!; seed!(1);

## parameters for neural network
nn = 100; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 1000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 1000; # maximum number of training iterations for opt2

dx = 0.05; # discretization size used for training
α_bc = 1.0f0 # weight on boundary conditions loss
Q_fpke = 0.1f0; # Q = σ^2
dt = 0.01; tEnd = 0.1;

# file location to save data
suff = string(activFunc);
expNum = 2;
saveFile = "data_rothe/vdp_exp$(expNum).jld2";
useGPU = true; if useGPU using CUDA end;
runExp = true;
runExp_fileName = "out_rothe/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "ts_vdp__PINN using Rothe's method with Grid training. 3 HL with $(nn) neurons in the hl and $(suff) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS. using GPU? $(useGPU). dx = $(dx). α_bc = $(α_bc). Q_fpke = $(Q_fpke). Not using ADAM, just LBFGS for $(maxOpt2Iters) iterations.
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
ρ(x) = exp(η(x[1],x[2]));

driftTerm = -(Differential(x1)(x2*exp(η(x1, x2))) + Differential(x2)(exp(η(x1, x2))*(x2*(1 - (x1^2)) - x1)))*(exp(η(x1, x2))^-1)
diffTerm1 = Differential(x2)(Differential(x2)(η(x1,x2))) 
diffTerm2 = abs2(Differential(x2)(η(x1,x2))) # works
diffTerm = Q_fpke/2*(diffTerm1 + diffTerm2); # diffusion term
pdeOpt = -driftTerm + diffTerm # full pde

pde_lhs = (η(xSym...) - dt/2*pdeOpt) ~ 0.0f0; # THIS IS NOT THE ACTUAL PDE
pde_rhs = (η(xSym...) + dt/2*pdeOpt) ~ 0.0f0; # THIS IS NOT THE ACTUAL PDE

## Domain
maxval = 4.0f0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval),
           x2 ∈ IntervalDomain(-maxval,maxval)];

# Boundary conditions
bcs = [ρ([-maxval,x2]) ~ 0.f0, ρ([maxval,x2]) ~ 0,
       ρ([x1,-maxval]) ~ 0.f0, ρ([x1,maxval]) ~ 0];

## Neural network
dim = 2 # number of dimensions
# chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1)); # 2 hls
chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1)); # 3 hls
# chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1)); # 4 hls

## Get get_th0
# # # Initial condition
μ0  = [0f0,0f0]; Σ0 = 0.1f0*1.0f0I(2); #gaussian 
ρ0_sym = pdf(MvNormal(μ0, Σ0),xSym);
ic_eqn = ρ(xSym) - ρ0_sym ~ 0.0f0;

initθ = DiffEqFlux.initial_params(chain);
if useGPU
    using CUDA
    CUDA.allowscalar(false)
    initθ = initθ |> gpu;
end
eltypeθ = eltype(initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(initθ);

strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = [x1, x2]
depvars = [η(x1,x2)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

_ic_loss_fn = NeuralPDE.build_loss_function(ic_eqn,indvars,depvars,phi,derivative,integral,chain,initθ,strategy);


train_domain_set, train_bound_set = NeuralPDE.generate_training_sets(domains, dx, [pde_lhs], bcs, eltypeθ, indvars, depvars) ;
if useGPU
    train_domain_set = train_domain_set |> gpu;
    train_bound_set = train_bound_set |> gpu;
end

using Statistics
ic_loss_fn =  (θ) -> mean(abs2,_ic_loss_fn(train_domain_set[1], θ));
@show ic_loss_fn(initθ)

ic_loss_fn_(θ, p) = ic_loss_fn(θ);

nSteps = 0;
cb0 = function (p, l)
    if any(isnan.(p))
        println("SOME PARAMETERS ARE NaN.")
    end

    global nSteps = nSteps + 1
    println("[$nSteps] Current loss is: $l")
    
    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[IC][$nSteps] Current loss is: $l \n")
        end;        
    end
    return false
end
println("Beginning optimization for initial condition...");
f0 = OptimizationFunction(ic_loss_fn_, GalacticOptim.AutoZygote())
prob0 = GalacticOptim.OptimizationProblem(f0, initθ)
# res0 = GalacticOptim.solve(prob0, opt1, cb=cb0, maxiters=maxOpt1Iters);
# prob0 = remake(prob0, u0=res0.minimizer)
res0 = GalacticOptim.solve(prob0, opt2, cb=cb0, maxiters=maxOpt2Iters);
println("Optimization for initial condition done.");
th0 = res0.minimizer;


θFull = []; # Initialize variable for storage
cuθFull = []; # Stores cu Arrays

push!(θFull, Array(th0));
push!(cuθFull, (th0));

_pde_lhs_fn = NeuralPDE.build_loss_function(pde_lhs,indvars,depvars,phi,derivative,integral,chain,initθ,strategy);
_pde_rhs_fn = NeuralPDE.build_loss_function(pde_rhs,indvars,depvars,phi,derivative,integral,chain,initθ,strategy);
_pde_loss_function(cord, θ1, θ0) = _pde_lhs_fn(cord, θ1) - _pde_rhs_fn(cord, θ0);

bc_indvars = NeuralPDE.get_argument(bcs, indvars, depvars);
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,phi,derivative,integral,chain,initθ,strategy,bc_indvars = bc_indvar) for (bc, bc_indvar) in zip(bcs, bc_indvars)]

bc_loss_functions = [NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy) for (loss, set) in zip(_bc_loss_functions, train_bound_set)]

bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions));
@show bc_loss_function_sum(initθ);

##
nT = Int(tEnd/dt) + 1
tR = LinRange(0.0, tEnd,Int(tEnd/dt)+1)

for (tInt, tVal) in enumerate(tR) 
    pde_loss_function = (θ1) -> mean(abs2,_pde_loss_function(train_domain_set[1], θ1, cuθFull[tInt]));
    @show pde_loss_function(initθ)

    function loss_function_(θ, p)
        return pde_loss_function(θ) + α_bc*bc_loss_function_sum(θ) 
    end
    @show loss_function_(initθ,0)

    ## set up GalacticOptim optimization problem
    f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
    prob = GalacticOptim.OptimizationProblem(f_, cuθFull[tInt])

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

        push!(PDE_losses, pde_loss_function(p))
        push!(BC_losses, bc_loss_function_sum(p))

        if runExp # if running job file
            open(runExp_fileName, "a+") do io
                write(io, "[t = $(tVal)][$nSteps] Current loss is: $l \n")
            end;
            
            # jldsave(saveFile; optParam=Array(p), PDE_losses, BC_losses);
        end
        return false
    end

    println("Calling GalacticOptim() at t =  $(tVal)");
    # res = GalacticOptim.solve(prob, opt1, cb=cb_, maxiters=maxOpt1Iters);
    # prob = remake(prob, u0=res.minimizer)
    res = GalacticOptim.solve(prob, opt2, cb=cb_, maxiters=maxOpt2Iters);

    push!(θFull, Array(res.minimizer));
    push!(cuθFull, (res.minimizer));
    println("Optimization done for t = $(tVal).");
end

# ## Save data
cd(@__DIR__);
if runExp
    jldsave(saveFile;optParams = θFull);
end