## Design a controller by solving the transient FPKE for the Van der Pol oscillator using baseline PINNs (large training set) and Rothe's method
# Grid strategy
cd(@__DIR__);
mkpath("out_cont_rothe");
mkpath("data_cont_rothe");
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra, Distributions

import Random:seed!; seed!(1);

## parameters for neural network
nn = 20; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 1000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 200; # maximum number of training iterations for opt2

dx = 0.1; # discretization size used for training
α_bc = 1.0f0 # weight on boundary conditions loss
Q_fpke = 0.0f0; # Q = σ^2
dt = 0.2; tEnd = 1.0;
μ0  = [1f0,2f0]; Σ0 = 0.5f0*1.0f0I(2); #gaussian 
A = 0.5f0*1.0f0I(2); # stable linear system

# file location to save data
suff = string(activFunc);
expNum = 5;
saveFile = "data_cont_rothe/vdp_exp$(expNum).jld2";
useGPU = false;
runExp = true;
runExp_fileName = "out_cont_rothe/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Designing a controller for ts_vdp__PINN using Rothe's method with Grid training. 2 HL with $(nn) neurons in the hl and $(suff) activation. using GPU? $(useGPU). dx = $(dx). α_bc = $(α_bc). Q_fpke = $(Q_fpke). dt = $(dt). tEnd = $(tEnd). Not using ADAM, just LBFGS for $(maxOpt2Iters) iterations. Model matching. μ0 = $(μ0). Σ0 = $(Σ0). A = $(A).
        Experiment number: $(expNum)\n")
    end
end

## set up the NeuralPDE framework using low-level API
@parameters x1, x2
@variables  Kc(..) # nonlinear control law

xSym = [x1;x2]

# Van der Pol Dynamics
f(x) = [x[2]; -x[1] + (1-x[1]^2)*x[2] + Kc(x...)];
g(x) = [0.0f0;1.0f0];

# Initial Condition 
nT = Int(tEnd/dt) + 1
tR = LinRange(0.0, tEnd,Int(tEnd/dt)+1)

μ = zeros(Float32,2,nT); μ[:,1] = μ0;
Σ = zeros(Float32,2,2,nT);  Σ[:,:,1] = Σ0;

# Linear System description

#
pde_eqn_lhs = 0.0f0;
for (tInt, tVal) in enumerate(tR[1:end-1]) 
    μ[:,tInt+1] = A*μ[:,tInt];
    Σ[:,:,tInt+1] = A*Σ[:,:,tInt]*A';

    ρ_sym1 = pdf(MvNormal(μ[:,tInt+1],Σ[:,:,tInt+1]), xSym);
    ρ_sym0 = pdf(MvNormal(μ[:,tInt],Σ[:,:,tInt]), xSym);

    F1 = f(xSym)*ρ_sym1; F0 = f(xSym)*ρ_sym0;
    G1 = 0.5f0*(g(xSym)*Q_fpke*g(xSym)')*ρ_sym1; G0 = 0.5f0*(g(xSym)*Q_fpke*g(xSym)')*ρ_sym0;

    drift1 = sum([Differential(xSym[i])(F1[i]) for i in 1:length(xSym)]);
    diff1 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G1[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
    pdeOpt1 = -drift1 #+ diff1;

    drift0 = sum([Differential(xSym[i])(F0[i]) for i in 1:length(xSym)]);
    diff0 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G0[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
    pdeOpt0 = -drift0 #+ diff0;

    global pde_eqn_lhs += abs2((ρ_sym1 - dt/2*pdeOpt1) - (ρ_sym0 + dt/2*pdeOpt0)) # error at time k squared here itself
end

pde_eqn = pde_eqn_lhs ~ 0.0f0; # FINAL PDE
placeholder_eqn = Kc(x1,x2) ~ 0.0f0; # just for generating training data 
println("PDE defined.")

## Neural network
chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1)); # 1 layer network
initθ,re  = Flux.destructure(chain)
# phi = (x,θ) -> re(θ)(Array(x))

## Domain
maxval = 4.0f0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval),
           x2 ∈ IntervalDomain(-maxval,maxval)];

# Boundary conditions
bcs = [Kc(-maxval,x2) ~ 0.f0, Kc(maxval,x2) ~ 0]; # placeholder, not used


## NeuralPDE set up
if useGPU
    using CUDA
    CUDA.allowscalar(false)
    initθ = initθ |> gpu;
end
flat_initθ = initθ; th0 = initθ;
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

strategy = NeuralPDE.GridTraining(dx);
derivative = NeuralPDE.get_numeric_derivative();

indvars = xSym
depvars = [Kc(xSym...)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

_pde_loss_function = NeuralPDE.build_loss_function(pde_eqn,indvars,depvars,phi,derivative,integral,chain,initθ,strategy);

train_domain_set, train_bound_set = NeuralPDE.generate_training_sets(domains, dx, [placeholder_eqn], bcs, eltypeθ, indvars, depvars) ;
if useGPU
    train_domain_set = train_domain_set |> gpu;
end

using Statistics
pde_loss_function = (θ) -> mean(_pde_loss_function(train_domain_set[1], θ));
@show pde_loss_function(initθ)

loss_function_(θ, p) = pde_loss_function(θ);
@show loss_function_(initθ, 0)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, initθ)

nSteps = 0;
PDE_losses = Float32[];
BC_losses = Float32[];
# IC_losses = Float32[];
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
    )

    push!(PDE_losses, pde_loss_function(p))

    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses, tR, μ, Σ, A)
    end
    return false
end

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt1, cb=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer)
res = GalacticOptim.solve(prob, opt2, cb=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

# ## Save data
cd(@__DIR__);
if runExp
    jldsave(saveFile;optParam = res.minimizer, PDE_losses, tR, μ, Σ, A);
end

##
th0 = initθ;
df(th) = Zygote.gradient(pde_loss_function, th)[1];
