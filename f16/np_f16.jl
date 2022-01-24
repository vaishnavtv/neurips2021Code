# using original neuralpde, no custom self-written loss function for pde - won't work because F16 Dynamics is incredibly complicated with interpolations and if conditions that mess up symbolic PDE generation. Self-written custom loss functions are not optimized for computation: second order derivative required in the diffusion term that is terribly slow using autodiff

println("Running quasi_baseline_f16 on CPU.")
## Solve the FPKE for the nonlinear F16 dynamics with LQR controller using baseline PINNs (large training set), 
cd(@__DIR__);
include("f16_controller.jl")
# include("NonlinearF16Model.jl")
# https://archive.siam.org/books/dc11/f16/Model.pdf for F16 Model Dynamics parameters

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, Symbolics, JLD2, ForwardDiff
using F16Model
using QuasiMonteCarlo

# using CUDA
# CUDA.allowscalar(false)
    
import Random:seed!;
seed!(1);

# parameters for neural network
nn = 50; # number of neurons in the hidden layers
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 1000; # maximum number of training iterations for opt2
α_bc = 0.01; # weight on boundary loss

nPtsPerMB = 1000;
nMB = 10;

expNum = 11;
saveFile = "data_ll_np/ll_quasi_f16$(expNum).jld2";
runExp = false; # flag to check if running batch file
runExp_fileName = ("out_ll_np/log$(expNum).txt");
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Running ll_quasi_f16_ using QuasiMonteCarlo strategy on CPU. pdelossfunction fixed. BC_losses coefficient: $(α_bc).
        $(nn) neurons in 2 hidden layers with $(activFunc) activation. $(maxOpt1Iters) iterations with ADAM $(opt1.eta) and then $(maxOpt2Iters) iterations with LBFGS. Updated quadrature strategy to use LatinHypercubeSample() alg. Resampling false. \nExperiment number: $(expNum).\n")
    end
end

## Dynamics Model
# Nominal Controller for Longitudinal F16Model trimmmed at specified altitude and velocity in 
# Trim vehicle at specified altitude and velocity
h0 = 10000; # ft
Vt0 = 500;   # ft/s

function f16Model_4x(x4, xbar, ubar, Kc)
    # nonlinear dynamics of 4-state model with stabilizing controller
    # x4 are the states, not perturbations
    xFull = Vector{Real}(undef, length(xbar))
    xFull .= xbar
    xFull[ind_x] .= Array(x4)# (x4)
    uFull = Vector{Real}(undef, length(ubar))
    uFull .= ubar
    u = (Kc * (Array(x4) .- xbar[ind_x])) # controller
    uFull[ind_u] .+= u
    # uFull[ind_u] = contSat(uFull[ind_u])
    xdotFull = F16Model.Dynamics(xFull, uFull)
    return xdotFull[ind_x] # return the 4 state dynamics
end

function contSat(u)
    # controller saturation
    if u[2] < -25.0
        u[2] = -25.0
    elseif u[2] > 25.0
        u[2] = 25.0
    end
    return u
end

@parameters xV, xα, xθ, xq
@variables η(..)

xSym = [xV, xα, xθ, xq]

f(x) = f16Model_4x(x, xbar, ubar, Kc) 

## PDE
function g(x)
    return Float32.(1.0I(4)); # Float32.([1.0; 1.0; 1.0; 1.0]) # diffusion in all states(?)
end

D_xV = Differential(xV);
D_xα = Differential(xα);
D_xθ = Differential(xθ);
D_xq = Differential(xq);
Q_fpke = 0.3f0 * 1.0I(4); # Q = σ^2

ρ(x) = exp(η(x...));
diffC = 1 / 2 * g(xSym) * Q_fpke * g(xSym)'; # diffusion coefficient D
F = f(xSym)*ρ(xSym);
G = diffC * ρ(xSym);

T1 = sum([Differential(xSym[i])(F[i]) for i in 1:length(xSym)]);
T2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);

Eqn = expand_derivatives(-T1+T2); # + dx*u(x1,x2)-1 ~ 0;
pdeOrig = simplify(Eqn/ρ(xSym)) ~ 0.0f0;
pde = pdeOrig;

## Domain
xV_min = 100;
xV_max = 1500;
xα_min = deg2rad(-20);
xα_max = deg2rad(40);
xθ_min = xα_min;
xθ_max = xα_max;
xq_min = -pi / 6;
xq_max = pi / 6;
domains = [
    xV ∈ IntervalDomain(xV_min, xV_max),
    xα ∈ IntervalDomain(xα_min, xα_max),
    xθ ∈ IntervalDomain(xθ_min, xθ_max),
    xq ∈ IntervalDomain(xq_min, xq_max),
];

## Grid discretization
dV = 100.0; dα = deg2rad(10); 
dθ = dα; dq = deg2rad(10);
dx = 0.5 * [dV; dα; dθ; dq]; # grid discretization in V (ft/s), α (rad), θ (rad), q (rad/s)


# Boundary conditions
bcs = [
    ρ([xV_min, xα, xθ, xq]) ~ 0.0f0,
    ρ([xV_max, xα, xθ, xq]) ~ 0.0f0,
    ρ([xV, xα_min, xθ, xq]) ~ 0.0f0,
    ρ([xV, xα_max, xθ, xq]) ~ 0.0f0,
    ρ([xV, xα, xθ_min, xq]) ~ 0.0f0,
    ρ([xV, xα, xθ_max, xq]) ~ 0.0f0,
    ρ([xV, xα, xθ, xq_min]) ~ 0.0f0,
    ρ([xV, xα, xθ, xq_max]) ~ 0.0f0,
];

## Neural network
dim = length(domains) # number of dimensions
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
# chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));;

initθ = DiffEqFlux.initial_params(chain)  # |> gpu;
eltypeθ = eltype(initθ)
@show eltypeθ;

parameterless_type_θ = DiffEqBase.parameterless_type(initθ);
@show parameterless_type_θ;

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = xSym
depvars = [η(xSym...)]

integral = NeuralPDE.get_numeric_integral(quasirandom_strategy, indvars, depvars, chain, derivative);

strategy = NeuralPDE.QuasiRandomTraining(nPtsPerMB;sampling_alg=UniformSample(), resampling=false,minibatch=nMB)


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
        quasirandom_strategy,
        bc_indvars=bc_indvar,
    ) for (bc, bc_indvar) in zip(bcs, bc_indvars)
]

pde_bounds, bcs_bounds = NeuralPDE.get_bounds(domains, [pde], bcs, eltypeθ, indvars, depvars, quasirandom_strategy)

## create loss functions
pde_loss_function = NeuralPDE.get_loss_function(
    _pde_loss_function,
    pde_bounds[1],
    eltypeθ,
    parameterless_type_θ,
    strategy,
);
@show pde_loss_function(initθ)

@show pde_loss_function_sum(initθ)
##
strategy_ = QuasiRandomTraining(quasirandom_strategy.bcs_points;
                                         sampling_alg=quasirandom_strategy.sampling_alg,
                                         resampling=quasirandom_strategy.resampling,
                                         minibatch=quasirandom_strategy.minibatch)

bc_loss_functions = [
    NeuralPDE.get_loss_function(loss, bound, eltypeθ, parameterless_type_θ, quasirandom_strategy) for (loss, bound) in zip(_bc_loss_functions, bcs_bounds)
]

bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
@show bc_loss_function_sum(initθ)
##
function loss_function_(θ, p)
    return pde_loss_function_sum(θ) + α_bc * bc_loss_function_sum(θ)
end
@show loss_function_(initθ, 0)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, initθ)

nSteps = 0;
PDE_losses = Float32[];
term1_losses = Float32[];
term2_losses = Float32[];
BC_losses = Float32[];
cb_ = function (p, l)
    global nSteps = nSteps + 1
    println("[$nSteps] Current loss is: $l")
    println(
        "Individual losses are: PDE loss:",
        pde_loss_function_sum(p),
        "; BC loss:",
        bc_loss_function_sum(p),
    )

    push!(PDE_losses, pde_loss_function_sum(p))
    push!(BC_losses,  bc_loss_function_sum(p))
    
    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses, BC_losses);
    end
    return false
end

println("Calling GalacticOptim()");
# res = GalacticOptim.solve(prob, opt1, cb=cb_, maxiters=maxOpt1Iters);
# prob = remake(prob, u0=res.minimizer)
# res = GalacticOptim.solve(prob, opt2, cb=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

## Save data
cd(@__DIR__);
if runExp
    jldsave(saveFile; optParam=Array(res.minimizer), PDE_losses, BC_losses);
end
