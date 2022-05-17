## Obtain a controller for f18 using PINNs
# terminal state pdf is fixed - using a thin gaussian about origin
cd(@__DIR__);
include("f18Dyn.jl")
mkpath("out_rhoConst")
mkpath("data_rhoConst")

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra, Distributions

import Random: seed!;
seed!(1);

## parameters for neural network
nn = 100; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

# parameters for rhoSS_desired
μ_ss = [0f0,0f0,0f0,0f0];
Σ_ss = 0.01f0*1.0f0I(4);

Q_fpke = 0.0f0; # Q = σ^2

# file location to save data
expNum = 1;
useGPU = false;
runExp = true;
saveFile = "data_rhoConst/exp$(expNum).jld2";
runExp_fileName = "out_rhoConst/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Generating a controller for f18 with desired ss distribution. 2 HL with $(nn) neurons in the hl and $(activFunc) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS. using GPU? $(useGPU). Q_fpke = $(Q_fpke). μ_ss = $(μ_ss). Σ_ss = $(Σ_ss). Not dividing equation by ρ.
        Experiment number: $(expNum)\n")
    end
end

## set up the NeuralPDE framework using low-level API
@parameters x1, x2, x3, x4
@variables Kc1(..), Kc2(..)

xSym = [x1; x2; x3; x4]

##
f18_xTrim = Float32.(1.0e+02*[3.500000000000000;-0.000000000000000;0.003540971009312;-0.000189987747098;0.000322083113778;0.000459982356948;0.006108652381980;0.003262466855376;0;]);
indX = [1;3;3;5];

f18_uTrim = Float32.(1.0e+04 *[-0.000000767698465;-0.000002371697733;-0.000007859275313;1.449999997030301;]);
indU = [3;4];

maskIndx = zeros(Float32,(length(f18_xTrim),length(indX)));
maskIndu = zeros(Float32,(length(f18_uTrim),length(indU)));
for i in 1:length(indX)
    maskIndx[indX[i],i] = 1f0;
    if i<=length(indU)
        maskIndu[indU[i],i] = 1f0;
    end
end

# F18 Dynamics
function f(xd)

    ud = [Kc1(xd...); Kc2(xd...)];

    xFull = f18_xTrim + maskIndx*xd;
    uFull = f18_uTrim + maskIndu*ud;

    xdotFull = f18Dyn(xFull, uFull)

    return (xdotFull[indX]) # return the 4 state dynamics

end

##
g(x::Vector) = [1.0f0; 1.0f0;1.0f0; 1.0f0] # diffusion vector needs to be modified

# PDE
ρSS_sym = pdf(MvNormal(μ_ss, Σ_ss),xSym);
F = f(xSym) * ρSS_sym;
G = 0.5f0 * (g(xSym) * Q_fpke * g(xSym)') * ρSS_sym;

T1 = sum([Differential(xSym[i])(F[i]) for i = 1:length(xSym)]);
T2 = sum([(Differential(xSym[i]) * Differential(xSym[j]))(G[i, j]) for i = 1:length(xSym),j = 1:length(xSym)]);

Eqn = expand_derivatives(-T1 + T2); # + dx*u(x1,x2)-1 ~ 0;
pde = simplify(Eqn) ~ 0.0f0;

println("PDE defined.")

## Domain
x1_min = -100f0; x1_max = 100f0;
x2_min = deg2rad(-10f0); x2_max = deg2rad(10f0);
x3_min = x2_min; x3_max = x2_max;
x4_min = deg2rad(-5f0); x4_max = deg2rad(5f0);
domains = [x1 ∈ IntervalDomain(x1_min, x1_max), x2 ∈ IntervalDomain(x2_min, x2_max), x3 ∈ IntervalDomain(x3_min, x3_max), x4 ∈ IntervalDomain(x4_min, x4_max),];

dx = [10f0; deg2rad(1f0); deg2rad(1f0); deg2rad(1f0);]; # discretization size used for training

# Boundary conditions
bcs = [Kc1(-100f0,x2,x3,x4) ~ 0.f0, Kc2(100f0,x2,x3,x4) ~ 0.f0]; # place holder, not really used


## Neural network set up
dim = 4 # number of dimensions
chain1 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain2 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain = [chain1, chain2];

initθ = DiffEqFlux.initial_params.(chain);
th0 = initθ;
th10= initθ[1];
th20= initθ[2];
if useGPU
    using CUDA
    CUDA.allowscalar(false)
    initθ = initθ |> gpu;
end
flat_initθ = reduce(vcat, initθ);
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);


strategy = NeuralPDE.GridTraining(dx);

indvars = xSym
depvars = [Kc1(xSym...), Kc2(xSym...)]

phi = NeuralPDE.get_phi.(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();
integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);
## Loss function

_pde_loss_function = NeuralPDE.build_loss_function(pde, indvars, depvars, phi, derivative, integral, chain, initθ, strategy);

train_domain_set, train_bound_set =
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);
if useGPU
    train_domain_set = train_domain_set |> gpu;
    train_bound_set = train_bound_set |> gpu;
end

using Statistics
pde_loss_function = (θ) -> mean(abs2,_pde_loss_function(train_domain_set[1], θ));
@show pde_loss_function(flat_initθ)

loss_function_(θ, p) =  pde_loss_function(θ)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, flat_initθ)

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

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt1, callback=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer);
res = GalacticOptim.solve(prob, opt2, callback=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

## Save data
if runExp
    jldsave(saveFile;optParam = Array(res.minimizer),PDE_losses);
end