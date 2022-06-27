## Obtain a controller for f18 using PINNs, approximating both ss ρ and u (ss1 from thesis)
# With diffusion, NAN issue
cd(@__DIR__);
include("f18Dyn.jl")
include("f18DynNorm.jl") # normalized state variable info
mkpath("out_ss1_cont")
mkpath("data_ss1_cont")

using NeuralPDE, Flux, ModelingToolkit, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra, Distributions, Optimization, GPUArrays

import Random: seed!;
seed!(1);

## parameters for neural network
nn = 100; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

Q_fpke = 0.1f0; # Q = σ^2

# parameters for rhoSS_desired
# μ_ss = [0f0,0f0,0f0,0f0] #.+ Array(f18_xTrim[indX]);
# Σ_ss = 0.01f0*Array(f18_xTrim[indX]).*1.0f0I(4);
μ_ss = An2*([0f0,0f0,0f0,0f0] .+ Array(f18_xTrim[indX])) + bn2;
Σ_ss = 0.01f0.*1.0f0I(4);

TMax = 50000f0; # maximum thrust
dStab_max = pi/3; # min, max values for δ_stab

dx = 0.1f0;
indU = [3,4]; # only using δ_stab for control

# file location to save data
expNum = 7;
useGPU = true;
runExp = true;
saveFile = "data_ss1_cont/exp$(expNum).jld2";
runExp_fileName = "out_ss1_cont/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Controller and ss distribution for the trimmed F18. 2 HL with $(nn) neurons in the hl and $(activFunc) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS. using GPU? $(useGPU). Q_fpke = $(Q_fpke). μ_ss = $(μ_ss). Σ_ss = $(Σ_ss). Using normalized variables between [-5,5], finding both δ_stab and T. Q_fpke = $(Q_fpke). dx = $(dx). Changed Σ_ss. Diffusion in α, deleting 0==0 equations from pdeDiff. Check Q_fpke.
        Experiment number: $(expNum)\n")
    end
end

## set up the NeuralPDE framework using low-level API
@parameters x1, x2, x3, x4
@variables η(..), Kc1(..), Kc2(..)

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

# F18 Dynamics
maskTrim = ones(Float32,length(f18_xTrim)); maskTrim[indX] .= 0f0;
function f(xn)
    # xd: perturbed state

    # xInp = xd .+ f18_xTrim[indX];
    # ud = Kc1(xInp[1],xInp[2],xInp[3],xInp[4]); 
    # # ud = [Kc1(xd[1],xd[2],xd[3],xd[4]); Kc2(xd...)]
    # # maskTrim = ones(Float32,length(f18_xTrim)); maskTrim[indX] .= 0f0;
    # xFull = f18_xTrim + maskIndx*xd; # perturbed state
    # uFull = [1f0;1f0;0f0;1f0].*f18_uTrim + maskIndu*ud; # utrim

    # xdotFull = f18Dyn(xFull, uFull)

    # return (xdotFull[indX]) # return the 4 state dynamics

    # normalized input to f18 dynamics (full dynamics)
    xi = An2Inv*(xn .- bn2); # x of 'i'nterest
    ui = [dStab_max*Kc1(xn...), TMax*Kc2(xn...)];

    xFull = maskTrim.*f18_xTrim + maskIndx*xi;
    uFull = [1f0;1f0;0f0;0f0].*f18_uTrim + maskIndu*ui; 

    xdotFull = f18Dyn(xFull, uFull)
    return An2*(xdotFull[indX]) # return the 4 state dynamics in normalized form
end

##
g(x::Vector) = [0.0f0; 1.0f0;0.0f0; 0.0f0] # diffusion in α

# PDE
ρ(x) = exp(η(x...));
F = f(xSym) * ρ(xSym);
G = 0.5f0 * (g(xSym) * Q_fpke * g(xSym)') * ρ(xSym);

# Drift Terms
driftTerms = ([Differential(xSym[i])(F[i]) for i = 1:length(xSym)]);
pdeDrift = driftTerms./(ρ(xSym)) .~ 0.0f0

# Diffusion
pdeDiff = Equation[]; # need to do the following to avoid NaNs
# (Differential(x1)(η(x1,x2,x3,x4)))^2 gives NaNs
# abs2(Differential(x1)(η(x1,x2,x3,x4))) is NaN-safe

for i in 1:4
    for j in 1:4
        if i!=j
            offDiagTerm = (Differential(xSym[i])*Differential(xSym[j]))(G[i,j])
            offDiagTerm = expand_derivatives(offDiagTerm)
            eqn = simplify(offDiagTerm/ρ(xSym), expand = true) ~ 0.0f0
            push!(pdeDiff, eqn);
        else
            eqn = 0.5f0*Q_fpke*g(xSym)[i]*(abs2(Differential(xSym[i])(η(xSym...))) + Differential(xSym[i])(Differential(xSym[i])(η(xSym...)))) ~ 0.0f0
            push!(pdeDiff, eqn);
        end
    end
end
## Delete unnecessary equations from pdeDiff
deleteat!(pdeDiff, [isequal(pdeDiff[i].lhs, 0.0f0) for i in 1:16])
println("PDE defined.")

## Domain
# x1_min = -100f0; x1_max = 100f0;
# x2_min = deg2rad(-10f0); x2_max = deg2rad(10f0);
# x3_min = x2_min ; x3_max = x2_max ;
# x4_min = deg2rad(-5f0); x4_max = deg2rad(5f0);

# All xi between 0 and 1
x1_min = vN2(f18_xTrim[indX[1]] - 100f0) ; x1_max = vN2(f18_xTrim[indX[1]] + 100f0) 
x2_min = alpN2(f18_xTrim[indX[2]] - deg2rad(10f0)) ; x2_max = alpN2(f18_xTrim[indX[2]] + deg2rad(10f0)) 
x3_min = thN2(f18_xTrim[indX[3]] - deg2rad(10f0)) ; x3_max = thN2(f18_xTrim[indX[3]] + deg2rad(10f0)) 
x4_min = qN2(f18_xTrim[indX[4]] + deg2rad(-5f0)) ; x4_max = qN2(f18_xTrim[indX[4]] + deg2rad(5f0)) 

domains = [x1 ∈ IntervalDomain(x1_min, x1_max), x2 ∈ IntervalDomain(x2_min, x2_max), x3 ∈ IntervalDomain(x3_min, x3_max), x4 ∈ IntervalDomain(x4_min, x4_max),];


# Boundary conditions
bcs = [η(x1_min,x2,x3,x4) ~ 0.f0, η(x1_max,x2,x3,x4) ~ 0.f0,
       η(x1,x2_min,x3,x4) ~ 0.f0, η(x1,x2_max,x3,x4) ~ 0.f0,
       η(x1,x2,x3_min,x4) ~ 0.f0, η(x1,x2,x3_max,x4) ~ 0.f0,
       η(x1,x2,x3,x4_min) ~ 0.f0, η(x1,x2_max,x3,x4_max) ~ 0.f0,];


## Neural network set up
dim = 4 # number of dimensions
chain1 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain2 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1, tanh));
chain3 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1, sigmoid));
chain = [chain1, chain2, chain3];

initθ = DiffEqFlux.initial_params.(chain);
if useGPU
    using CUDA
    CUDA.allowscalar(false)
    initθ = initθ |> gpu;
end 
flat_initθ = reduce(vcat,initθ); 
th0 = flat_initθ;
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);

indvars = xSym
depvars = [η(xSym...), Kc1(xSym...), Kc2(xSym...)]

phi = NeuralPDE.get_phi.(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();
integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

tx = cu(f18_xTrim[indX]);

## Loss function
println("Defining pde loss function for drift and diffusion terms.")
_pde_loss_functions_drift = [NeuralPDE.build_loss_function(pde_i, indvars, depvars, phi, derivative, integral, chain, initθ, strategy) for pde_i in pdeDrift];
@show [fn(tx, th0) for fn in _pde_loss_functions_drift]

_pde_loss_functions_diff = [NeuralPDE.build_loss_function(pde_i, indvars, depvars, phi, derivative, integral, chain, initθ, strategy) for pde_i in pdeDiff];
@show [fn(tx, th0) for fn in _pde_loss_functions_diff]


_pde_loss_function(cord, θ) =  sum([fn(cord, θ) for fn in _pde_loss_functions_drift]) .- sum([fn(cord, θ) for fn in _pde_loss_functions_diff])
@show _pde_loss_function(tx, th0) 

bc_indvars = NeuralPDE.get_argument(bcs, indvars, depvars);
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,phi,derivative,integral,chain,initθ,strategy,bc_indvars = bc_indvar) for (bc, bc_indvar) in zip(bcs, bc_indvars)]

# Domain and training sets
train_domain_set, train_bound_set =
    NeuralPDE.generate_training_sets(domains, dx, pdeDrift, bcs, eltypeθ, indvars, depvars);
if useGPU
    train_domain_set = train_domain_set |> gpu;
    train_bound_set = train_bound_set |> gpu;
end

using Statistics
pde_loss_function = (θ) -> mean(abs2,_pde_loss_function(cu(train_domain_set[1]), θ));
@show pde_loss_function(flat_initθ)

bc_loss_functions = [NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy) for (loss, set) in zip(_bc_loss_functions, train_bound_set)]

bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
@show bc_loss_function_sum(flat_initθ)

## Desired ss loss function
using Distributions
ρSS_sym = pdf(MvNormal(μ_ss, Σ_ss),xSym)
rhoSSEq = ρ(xSym) - ρSS_sym ~ 0.0f0;

_rhoSS_loss_function = NeuralPDE.build_loss_function(rhoSSEq,indvars,depvars,phi,derivative,integral,chain,initθ,strategy);

rhoSS_loss_function = NeuralPDE.get_loss_function(_rhoSS_loss_function,train_domain_set[1], eltypeθ, parameterless_type_θ, strategy);
@show rhoSS_loss_function(flat_initθ);

## Composite loss
loss_function_(θ, p) =  pde_loss_function(θ) + bc_loss_function_sum(θ) + rhoSS_loss_function(θ)
@show loss_function_(flat_initθ, 0)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, Optimization.AutoZygote())
prob = Optimization.OptimizationProblem(f_, flat_initθ)

nSteps = 0;
PDE_losses = Float32[];
BC_losses = Float32[];
rhoSS_losses = Float32[];
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
        ", rhoSS loss:",
        rhoSS_loss_function(p),
    )

    push!(PDE_losses, pde_loss_function(p))
    push!(BC_losses, bc_loss_function_sum(p))
    push!(rhoSS_losses, rhoSS_loss_function(p))

    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses, BC_losses, rhoSS_losses);
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
    jldsave(saveFile;optParam = Array(res.minimizer),PDE_losses, BC_losses, rhoSS_losses);
end