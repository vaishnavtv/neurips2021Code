## Obtain a controller for f18 using PINNs using Rothe's method
cd(@__DIR__);
include("../f18/f18Dyn.jl")
include("../f18/f18DynNorm.jl") # normalized state variable info
mkpath("out_ts_cont")
mkpath("data_ts_cont")

using NeuralPDE, Flux, ModelingToolkit, Optimization, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra, Distributions

import Random: seed!;
seed!(1);

## parameters for neural network
nn = 100; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

dt = 0.2f0; tEnd = 1.0f0;
μ0  = An2*([0f0,0f0,0f0,0f0] .+ Array(f18_xTrim[indX])) + bn2; # full dynamics
Σ0 = 1f0*1.0f0I(4); #gaussian
A = -0.5f0*1.0f0I(4); # stable linear system

Q_fpke = 0.0f0; # Q = σ^2
dx = 0.1f0;
TMax = 50000f0; # maximum thrust
dStab_max = pi/3; # min, max values for δ_stab

# file location to save data
expNum = 2;
useGPU = false;
runExp = true;
saveFile = "data_ts_cont/exp$(expNum).jld2";
runExp_fileName = "out_ts_cont/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Designing a controller for ts_f18 using Rothe's method. 2 HL with $(nn) neurons in the hl and $(activFunc) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS. using GPU? $(useGPU). Q_fpke = $(Q_fpke). μ0 = $(μ0). Σ0 = $(Σ0). A = $(A). Model-matching. Added dStab_max and TMax with tanh and sigmoid activation functions on output for δ_stab and Thrust. Full dynamics. Initial Condition: Gaussian about trim point. Not using GPU. 
        Experiment number: $(expNum)\n")
    end
end

## set up the NeuralPDE framework using low-level API
@parameters x1, x2, x3, x4
@variables Kc1(..), Kc2(..)

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
maskK = Float32.(maskIndu*Kc_lqr) # masking linear controller in perturbation
# F18 Dynamics
maskTrim = ones(Float32,length(f18_xTrim)); maskTrim[indX] .= 0f0;
function f(xn)

    # # ud = Kc1(xd[1],xd[2],xd[3],xd[4]); 
    # ud = [Kc1(xd[1],xd[2],xd[3],xd[4]); Kc2(xd[1],xd[2],xd[3],xd[4])];

    # # tx = ((maskIndx)*xd); tu = ((maskIndu)*ud);
    # # xFull = Vector{Real}(undef, 9);
    # # uFull = Vector{Real}(undef, 4);
    # # for i in 1:9
    # #     xFull[i] = f18_xTrim[i] + tx[i];
    # # end 
    # # for i in 1:4
    # #     uFull[i] = f18_uTrim[i] + tu[i];
    # # end 
    # # perturbation about trim point
    # xFull = f18_xTrim + maskIndx*xd; 
    # maskTrim = ones(Float32,length(f18_xTrim)); maskTrim[indX] .= 0f0;
    # # xFull = maskTrim.*f18_xTrim + maskIndx*xd; 
    # uFull = [1f0;1f0;0f0;1f0].*f18_uTrim + maskIndu*ud;

    # xdotFull = f18Dyn(xFull, uFull)
    # # xdotFull = xFull;

    # return (xdotFull[indX]) # return the 4 state dynamics

    # normalized input to f18 dynamics (full dynamics)
    xi = An2Inv*(xn .- bn2); # x of 'i'nterest
    ui = [dStab_max*Kc1((xn)...), TMax*Kc2((xn)...)];

    xFull = maskTrim.*f18_xTrim + maskIndx*xi;
    uFull = [1f0;1f0;0f0;0f0].*f18_uTrim + maskIndu*ui; 

    xdotFull = f18Dyn(xFull, uFull)
    return An2*(xdotFull[indX]) # return the 4 state dynamics in normalized form

    # normalized input to f18 dynamics (xn is perturbation)
    # xi = An3Inv*(xn .- bn3); # x of 'i'nterest,, perturbation
    # ui = [dStab_max*Kc1((xn)...), TMax*Kc2((xn)...)];

    # xFull = f18_xTrim + maskIndx*xi;
    # uFull = f18_uTrim .+ maskIndu*ui .+ (maskK)*xi;

    #  xdotFull = f18Dyn(xFull, uFull)
    # return An3*(xdotFull[indX]) # return the 4 state dynamics in normalized form
end

# Initial Condition 
nT = Int(tEnd/dt) + 1
tR = LinRange(0.0, tEnd,Int(tEnd/dt)+1)

μ = zeros(Float32,4,nT); μ[:,1] = μ0;
Σ = zeros(Float32,4,4,nT);  Σ[:,:,1] = Σ0;

pde_eqn_lhs = 0.0f0;
eqns_lhs = []; eqns = [];
for (tInt, tVal) in enumerate(tR[1:end-1]) 
    μ[:,tInt+1] = A*μ[:,tInt];
    Σ[:,:,tInt+1] = A*Σ[:,:,tInt]*A';

    ρ_sym1 = pdf(MvNormal(μ[:,tInt+1],Σ[:,:,tInt+1]), xSym);
    ρ_sym0 = pdf(MvNormal(μ[:,tInt],Σ[:,:,tInt]), xSym);

    F1 = f(xSym)*ρ_sym1; F0 = f(xSym)*ρ_sym0;

    drift1 = sum([Differential(xSym[i])(F1[i]) for i in 1:length(xSym)]);
    pdeOpt1 = -drift1;

    drift0 = sum([Differential(xSym[i])(F0[i]) for i in 1:length(xSym)]);
    pdeOpt0 = -drift0;

    push!(eqns_lhs, abs2((ρ_sym1 - dt/2*pdeOpt1) - (ρ_sym0 + dt/2*pdeOpt0)))
    push!(eqns, eqns_lhs[tInt] ~ 0.0f0)
    global pde_eqn_lhs += abs2((ρ_sym1 - dt/2*pdeOpt1) - (ρ_sym0 + dt/2*pdeOpt0)) # error at time k squared here itself
end

pde_eqn = pde_eqn_lhs ~ 0.0f0; # FINAL PDE
placeholder_eqn = Kc1(xSym...) ~ 0.0f0; # just for generating training data 

## Domain
# All xi between [bd,bd]
x1_min = vN2(f18_xTrim[indX[1]] - 100f0) ; x1_max = vN2(f18_xTrim[indX[1]] + 100f0) 
x2_min = alpN2(f18_xTrim[indX[2]] - deg2rad(10f0)) ; x2_max = alpN2(f18_xTrim[indX[2]] + deg2rad(10f0)) 
x3_min = thN2(f18_xTrim[indX[3]] - deg2rad(10f0)) ; x3_max = thN2(f18_xTrim[indX[3]] + deg2rad(10f0))
x4_min = qN2(f18_xTrim[indX[4]] + deg2rad(-5f0)) ; x4_max = qN2(f18_xTrim[indX[4]] + deg2rad(5f0)) 

domains = [x1 ∈ IntervalDomain(x1_min, x1_max), x2 ∈ IntervalDomain(x2_min, x2_max), x3 ∈ IntervalDomain(x3_min, x3_max), x4 ∈ IntervalDomain(x4_min, x4_max),];

# Boundary conditions
bcs = [Kc1(x1_min,x2,x3,x4) ~ 0.f0, Kc2(100f0,x2,x3,x4) ~ 0.f0]; # place holder, not really used

## Neural network set up
dim = 4 # number of dimensions
chain1 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1, tanh));
chain2 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1, sigmoid));
chain = [chain1, chain2];

initθ = DiffEqFlux.initial_params.(chain);
if useGPU
    using CUDA
    CUDA.allowscalar(false)
    initθ = initθ |> gpu;
    th10= initθ[1];
    th20= initθ[2];
end 
flat_initθ = reduce(vcat, initθ); 
th0 = flat_initθ;
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);


strategy = NeuralPDE.GridTraining(dx);

indvars = xSym
depvars = [Kc1(xSym...), Kc2(xSym...)]

phi = NeuralPDE.get_phi.(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();
integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

train_domain_set, train_bound_set =
    NeuralPDE.generate_training_sets(domains, dx, [placeholder_eqn], bcs, eltypeθ, indvars, depvars);
if useGPU
    train_domain_set = train_domain_set |> gpu;
end
@show size(train_domain_set[1])

## Loss function
println("Defining loss function for each term.")
_pde_loss_functions = [NeuralPDE.build_loss_function(eq,indvars,depvars,phi,derivative,integral,chain,initθ,strategy) for eq in eqns];
# @show [fn(train_domain_set[1][:,1], th0) for fn in _pde_loss_functions]

pde_loss_functions = [(θ) -> mean(fn(train_domain_set[1], θ)) for fn in _pde_loss_functions];

pde_loss_function = θ -> sum(map(l -> l(θ), pde_loss_functions)); # sum of all losses
@show pde_loss_function(th0)

loss_function_(θ, p) =  pde_loss_function(θ)

## set up Optimization optimization problem
f_ = OptimizationFunction(loss_function_, Optimization.AutoZygote())
prob = Optimization.OptimizationProblem(f_, flat_initθ)

nSteps = 0;
PDE_losses = Float32[];
cb_ = function (p, l)
    if any(isnan.(p))
        println("SOME PARAMETERS ARE NaN.")
    end

    global nSteps = nSteps + 1
    println("[$nSteps] Current loss is: $l")

    push!(PDE_losses, l)

    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses);
    end
    return false
end

println("Calling Optimization()");
res = Optimization.solve(prob, opt1, callback=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer);
res = Optimization.solve(prob, opt2, callback=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

## Save data
if runExp
    jldsave(saveFile;optParam = Array(res.minimizer),PDE_losses);
end
