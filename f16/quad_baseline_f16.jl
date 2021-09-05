println("Running quad_baseline_f16 on CPU.")
@show Threads.nthreads()
## Solve the FPKE for the nonlinear F16 dynamics with LQR controller using baseline PINNs (large training set)
cd(@__DIR__);
include("f16_controller.jl")
# include("NonlinearF16Model.jl")
# https://archive.siam.org/books/dc11/f16/Model.pdf for F16 Model Dynamics parameters

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, Symbolics, JLD2, ForwardDiff
using F16Model
using Quadrature, Cubature

# using CUDA
# CUDA.allowscalar(false)
    
import Random: seed!;
seed!(1);

# parameters for neural network
nn = 50; # number of neurons in the hidden layers
activFunc = tanh; # activation function
opt1 = ADAM(5e-3); # primary optimizer used for training
maxOpt1Iters = 100; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 100; # maximum number of training iterations for opt2

expNum = 4;
saveFile = "dataQuad/quad_baseline_f16_ADAM_$(expNum).jld2";
runExp = true; # flag to check if running batch file
if runExp
    open("outQuad/log$(expNum).txt", "a+") do io
        write(io, "Running baselin_f16 using Quadrature strategy on CPU. $(nn) neurons in the 2 hidden layers. $(maxOpt1Iters) iterations with ADAM (5e-3) and then $(maxOpt2Iters) iterations with LBFGS. \nExperiment number: $(expNum).\n")
    end
end
# end;
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
    xFull[ind_x] .= Array(x4)#(x4)
    uFull = Vector{Real}(undef, length(ubar))
    uFull .= ubar
    u = (Kc * (Array(x4) .- xbar[ind_x])) # controller
    # u[2] = rad2deg(u[2]);
    # @show u
    uFull[ind_u] .+= u
    # @show uFull[ind_u]
    uFull[ind_u] = contSat(uFull[ind_u])
    xdotFull = F16Model.Dynamics(xFull, uFull)
    # xdotFull = F16ModelDynamics_sym(xFull, uFull)
    # return u
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
    return Float32.([1.0; 1.0; 1.0; 1.0]) # diffusion in all states(?)
end

D_xV = Differential(xV);
D_xα = Differential(xα);
D_xθ = Differential(xθ);
D_xq = Differential(xq);
Q = 0.3f0; # Q = σ^2

ρ(x) = exp(η(x...));
G = 0.5f0 * (g(xSym) * Q * g(xSym)') * ρ(xSym);
T2 = sum([
    (Differential(xSym[i]) * Differential(xSym[j]))(G[i, j]) for i = 1:length(xSym), j = 1:length(xSym)
]);

Eqn = expand_derivatives(T2);
pdeOrig = simplify(Eqn) ~ 0.0f0; # only term2, term1 coming from self-written function since it cannot be expressed symbolically atm
pde = ((0.15f0(Differential(xq)(η(xV, xα, xθ, xq))^2) + 0.15f0(Differential(xα)(η(xV, xα, xθ, xq))^2) + 0.15f0(Differential(xθ)(η(xV, xα, xθ, xq))^2) + 0.15f0(Differential(xV)(η(xV, xα, xθ, xq))^2))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xθ)(Differential(xq)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xθ)(Differential(xθ)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xV)(Differential(xV)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xα)(Differential(xα)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xα)(Differential(xq)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xq)(Differential(xV)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xq)(Differential(xα)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xθ)(Differential(xα)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xq)(Differential(xq)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xα)(Differential(xθ)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xq)(Differential(xθ)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xV)(Differential(xθ)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xV)(Differential(xq)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xα)(Differential(xV)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xθ)(Differential(xV)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xV)(Differential(xα)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.3f0Differential(xq)(η(xV, xα, xθ, xq))*Differential(xV)(η(xV, xα, xθ, xq))*exp(η(xV, xα, xθ, xq)) + 0.3f0Differential(xq)(η(xV, xα, xθ, xq))*Differential(xα)(η(xV, xα, xθ, xq))*exp(η(xV, xα, xθ, xq)) + 0.3f0Differential(xα)(η(xV, xα, xθ, xq))*Differential(xV)(η(xV, xα, xθ, xq))*exp(η(xV, xα, xθ, xq)) + 0.3f0Differential(xθ)(η(xV, xα, xθ, xq))*Differential(xα)(η(xV, xα, xθ, xq))*exp(η(xV, xα, xθ, xq)) + 0.3f0Differential(xV)(η(xV, xα, xθ, xq))*Differential(xθ)(η(xV, xα, xθ, xq))*exp(η(xV, xα, xθ, xq)) + 0.3f0Differential(xq)(η(xV, xα, xθ, xq))*Differential(xθ)(η(xV, xα, xθ, xq))*exp(η(xV, xα, xθ, xq)))~0.0f0; #*(exp(η(xV, xα, xθ, xq))^-1) ~ 0.0f0;

# pde = exp(η(xV, xα, xθ, xq))*(0.15f0(Differential(xα)(η(xV, xα, xθ, xq))^2) + 0.15f0(Differential(xV)(η(xV, xα, xθ, xq))^2)) + 0.15f0Differential(xV)(Differential(xV)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xα)(Differential(xα)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xα)(Differential(xV)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.15f0Differential(xV)(Differential(xα)(η(xV, xα, xθ, xq)))*exp(η(xV, xα, xθ, xq)) + 0.3f0Differential(xα)(η(xV, xα, xθ, xq))*Differential(xV)(η(xV, xα, xθ, xq))*exp(η(xV, xα, xθ, xq)) ~ 0.0f0;
## Domain
xV_min = 100;
xV_max = 1500;
xα_min = deg2rad(-20);
xα_max = deg2rad(40);
xθ_min = xα_min;
xθ_max = xα_max;
xq_min = -pi/6;
xq_max = pi/6;
domains = [
    xV ∈ IntervalDomain(xV_min, xV_max),
    xα ∈ IntervalDomain(xα_min, xα_max),
    xθ ∈ IntervalDomain(xθ_min, xθ_max),
    xq ∈ IntervalDomain(xq_min, xq_max),
];

## Grid discretization
dV = 100.0; dα = deg2rad(10); 
dθ = dα; dq = deg2rad(10);
dx = 0.5*[dV; dα; dθ; dq]; # grid discretization in V (ft/s), α (rad), θ (rad), q (rad/s)


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
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubatureJLh(),
                                                    reltol=1e-3,abstol=1e-3,
                                                    maxiters =50, batch=100)
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));;

initθ = DiffEqFlux.initial_params(chain) #|> gpu;
eltypeθ = eltype(initθ)
@show eltypeθ;

parameterless_type_θ = DiffEqBase.parameterless_type(initθ);
@show parameterless_type_θ;

strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = xSym
depvars = [η(xSym...)]

integral = NeuralPDE.get_numeric_integral(quadrature_strategy, indvars, depvars, chain, derivative);

##
_pde_term2_loss_function = NeuralPDE.build_loss_function(
    pde,
    indvars,
    depvars,
    phi,
    derivative,
    integral,
    chain,
    initθ,
    quadrature_strategy,
);

# bc_indvars = NeuralPDE.get_variables(bcs, indvars, depvars);
bc_indvars = NeuralPDE.get_argument(bcs,indvars,depvars);
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
        quadrature_strategy,
        bc_indvars = bc_indvar,
    ) for (bc, bc_indvar) in zip(bcs, bc_indvars)
]

train_domain_set, train_bound_set =
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);
train_domain_set_cpu = Array(train_domain_set[1])
nTrainDomainSet = size(train_domain_set[1],2)

pde_bounds, bcs_bounds = NeuralPDE.get_bounds(domains,[pde],bcs,eltypeθ,indvars,depvars,quadrature_strategy)
pde_bounds = pde_bounds #|> gpu;
lbs, ubs = pde_bounds;
## create loss functions

function term1_pde_loss_function(θ)
    # custom self-written pde loss function for term1. term2 can be symbolically expressed and too time-consuming for self-written hessian
    ρFn(y) = exp(first(Array(phi(y, θ)))); # phi is the NN representing η

    fxρ(y) = f(y)*ρFn(y);
    gxρ(y) = 0.5 * (g(y) * Q * g(y)') * ρFn(y);
    
    function term1(y)
        tmp = ForwardDiff.jacobian(fxρ, y)
        return sum(diag(tmp))
    end
    # function term2(y) # takes too long to compute
    #     tmp = 0;
    #     for j = 1:4
    #         for i = 1:4
    #             g2(x) = gxρ(x)[i,j];
    #             g2_ij(x) = ForwardDiff.hessian(g2, x)[i,j]
    #             # @show g2_ij(y)
    #             # sleep(100);
    #             # g2_ij_arr = Array(g2_ij(y));
    #             tmp += g2_ij(y)#[i,j]
    #         end
    #     end
    #     return tmp
    # end
    pdeErr(y) = (-term1(y))^2;# + term2(y))^2; # pdeErr evaluated at state y
    
    tmp = Float32(sum(pdeErr(train_domain_set_cpu[:,i]) for i in 1:nTrainDomainSet)/nTrainDomainSet) #mean squared error
    return tmp
end
@show term1_pde_loss_function(initθ)

term2_pde_loss_function = NeuralPDE.get_loss_function(
    _pde_term2_loss_function,
    (lbs[1]), (ubs[1]),
    eltypeθ,
    parameterless_type_θ,
    quadrature_strategy,
);
@show term2_pde_loss_function(initθ)
pde_loss_function_custom(θ) = term1_pde_loss_function(θ) + term2_pde_loss_function(θ); # full pde loss function (custom written)
@show pde_loss_function_custom(initθ)

lbs, ubs = bcs_bounds;
bc_loss_functions = [
    NeuralPDE.get_loss_function(loss, lb, ub, eltypeθ, parameterless_type_θ, quadrature_strategy) for
    (loss, lb, ub) in zip(_bc_loss_functions, lbs, ubs)
]

bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
@show bc_loss_function_sum(initθ)

function loss_function_(θ, p)
    return pde_loss_function_custom(θ) + bc_loss_function_sum(θ)
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
        pde_loss_function_custom(p),
        "; term1 loss:",
        term1_pde_loss_function(p),
        "; term2 loss:",
        term2_pde_loss_function(p),
        "; BC loss:",
        bc_loss_function_sum(p),
    )

    push!(PDE_losses, pde_loss_function_custom(p))
    push!(BC_losses, bc_loss_function_sum(p))
    push!(term1_losses, term1_pde_loss_function(p))
    push!(term2_losses, term2_pde_loss_function(p))

    if runExp # if running job file
        open("outQuad/log$(expNum).txt", "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam = Array(p), PDE_losses, BC_losses, term1_losses, term2_losses); ## save for checking.
    end
    return false
end

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt1, cb = cb_, maxiters = maxOpt1Iters);
prob = remake(prob, u0 = res.minimizer)
res = GalacticOptim.solve(prob, opt2, cb = cb_, maxiters = maxOpt2Iters);
println("Optimization done.");

## Save data
cd(@__DIR__);
if runExp
    jldsave(saveFile; optParam = Array(res.minimizer), PDE_losses, BC_losses, term1_losses, term2_losses);
end
