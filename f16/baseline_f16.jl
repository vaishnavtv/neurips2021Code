## Solve the FPKE for the Van der Pol Rayleigh oscillator using baseline PINNs (large training set)
cd(@__DIR__);
include("f16_controller.jl")
# include("NonlinearF16Model.jl")

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, Symbolics, JLD2
using F16Model

# parameters for neural network
nn = 100; # number of neurons in the hidden layers
activFunc = tanh; # activation function
maxOptIters = 10; # maximum number of training iterations
# opt = Optim.LBFGS(); # Optimizer used for training
opt = ADAM(1e-3); 

##
# Nominal Controller for Longitudinal F16Model trimmmed at specified altitude and velocity in 
# Trim vehicle at specified altitude and velocity
h0 = 10000; # ft
Vt0 = 500;   # ft/s

function f16Model_4x(x4, xbar, ubar, Kc)
    # nonlinear dynamics of 4-state model with stabilizing controller
    # x4 are the states, not perturbations
    xFull = Vector{Real}(undef, length(xbar))
    xFull .= xbar
    xFull[ind_x] .= x4#(x4)
    uFull = Vector{Real}(undef, length(ubar))
    uFull .= ubar
    # @show ubar[ind_u]
    u = (Kc * (x4 .- xbar[ind_x])) # controller
    # @show u
    uFull[ind_u] .+= u
    xdotFull = F16Model.Dynamics(xFull, uFull)
    # xdotFull = F16ModelDynamics_sym(xFull, uFull)
    return xdotFull[ind_x] # return the 4 state dynamics
end

@parameters xV, xα, xθ, xq
@variables η(..)

xSym = [xV, xα, xθ, xq]

f(x) = f16Model_4x(x, xbar, ubar, Kc) 

## PDE
function g(x)
    return [1.0; 0.0; 0.0; 0.0] # noise 
end

D_xV = Differential(xV);
D_xα = Differential(xα);
D_xθ = Differential(xθ);
D_xq = Differential(xq);
Q = 0.3; # Q = σ^2

ρ(x) = exp(η(x[1], x[2], x[3], x[4]));

pde = D_xV(ρ(xSym)) ~ 0.0f0; # placeholder pde, does not get used. Created only for NeuralPDE purposes.

## Domain
xV_min = 300;
xV_max = 1000;
xα_min = deg2rad(-5);
xα_max = pi / 6;
xθ_min = xα_min;
xθ_max = xα_max;
xq_min = 0;
xq_max = 1.0;
domains = [
    xV ∈ IntervalDomain(xV_min, xV_max),
    xα ∈ IntervalDomain(xα_min, xα_max),
    xθ ∈ IntervalDomain(xθ_min, xθ_max),
    xq ∈ IntervalDomain(xq_min, xq_max),
];

# Grid discretization
dV = 100.0; dα = deg2rad(10.0); dθ = deg2rad(10.0); dq =  0.5;
dx = [dV; dα; dθ; dq]; # grid discretization in V (ft/s), α (deg), θ (deg), q (rad/s)


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
dim = 4 # number of dimensions
nn = 10; activFunc = tanh;
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));#|> gpu;

initθ = DiffEqFlux.initial_params(chain);#|> gpu;
eltypeθ = eltype(initθ)

parameterless_type_θ = DiffEqBase.parameterless_type(initθ);

strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = xSym
depvars = [η]

##
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
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);# |> gpu;
train_domain_set = train_domain_set #|> gpu

function pde_loss_function_custom(θ)
    # custom self-written pde loss function. need to check again.
    ρFn(y) = exp(phi(y, θ)[1]); # phi is the NN

    fxρ(y) = f(y)*ρFn(y);
    gxρ(y) = 0.5 * (g(y) * Q * g(y)') * ρFn(y);
    
    function term1(y)
        tmp = ForwardDiff.jacobian(fxρ, y)
        # tmp = ForwardDiff.gradient(ρFn, y)
        # return sum(tmp)
        return sum(diag(tmp))
    end
    function term2(y)
        tmp = 0;
        for j = 1:4
            for i = 1:4
                g2(x) = gxρ(x)[i,j];
                # g2_i(y) = ForwardDiff.gradient(g2, y)[i]
                # g2_ij(y) = ForwardDiff.gradient(g2_i,y)[j]
                g2_ij(x) = ForwardDiff.hessian(g2, x)[i,j]
                tmp += g2_ij(y)
            end
        end
        return tmp
    end
    pdeErr(y) = -term1(y) + term2(y); # pdeErr evaluated at state y
    
    nTrainDomainSet = size(train_domain_set[1],2)
    tmp = Float32(sum(pdeErr(train_domain_set[1][:,i])^2 for i in 1:nTrainDomainSet)/nTrainDomainSet) # mean squared error
    return tmp
end
@show pde_loss_function_custom(initθ)

# gxρ(xbar[ind_x])
# g2(y) = gxρ(y)[1,1];
# ForwardDiff.hessian(g2, xbar[ind_x])
# sum(term2(train_domain_set[1][:,i]) for i in 1:96)
# sum(term1(train_domain_set[1][:,i]) for i in 1:96)
# sum(ρFn(train_domain_set[1][:,i]) for i in 1:96)

bc_loss_functions = [
    NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy) for
    (loss, set) in zip(_bc_loss_functions, train_bound_set)
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
BC_losses = Float32[];
cb_ = function (p, l)
    global nSteps = nSteps + 1
    println("[$nSteps] Current loss is: $l")
    println(
        "Individual losses are: PDE loss:",
        pde_loss_function_custom(p),
        ", BC loss:",
        bc_loss_function_sum(p),
    )

    push!(PDE_losses, pde_loss_function_custom(p))
    push!(BC_losses, bc_loss_function_sum(p))
    return false
end

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt, cb = cb_, maxiters = maxOptIters);
println("Optimization done.");

## Save data
cd(@__DIR__);
# jldsave(saveFile; optParam = Array(res.minimizer), PDE_losses, BC_losses);
