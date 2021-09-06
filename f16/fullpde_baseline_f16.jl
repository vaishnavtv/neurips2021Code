println("Running baseline_f16 with pdelossfunction that runs on CPU arrays.")
@show Threads.nthreads()
## Solve the FPKE for the nonlinear F16 dynamics with LQR controller using baseline PINNs (large training set)
cd(@__DIR__);
include("f16_controller.jl")
# include("NonlinearF16Model.jl")
# https://archive.siam.org/books/dc11/f16/Model.pdf for F16 Model Dynamics parameters

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, Symbolics, JLD2, ForwardDiff
using F16Model

using CUDA
CUDA.allowscalar(false)
    
import Random: seed!;
seed!(1);

# parameters for neural network
nn = 50; # number of neurons in the hidden layers
activFunc = tanh; # activation function
maxOptIters = 2000; # maximum number of training iterations
# opt = Optim.LBFGS(); # Optimizer used for training
opt = ADAM(1e-3); 
expNum = 1;
saveFile = "data_fullpde/fullpde_f16_cpu_$(expNum).jld2";
runExp = true; # flag to check if running batch file
if runExp
    open("out_fullpde/log$(expNum).txt", "a+") do io
        write(io, "Running everything on CPU with $(nn) neurons in 2 hidden layers for $(maxOptIters) iterations. Using just ADAM with lr 1e-3. Discretization size updated (~60k points). \n")
    end;
end
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
    return (xdotFull[ind_x]) # return the 4 state dynamics
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
Q_fpke = 0.3f0; # Q = σ^2

ρ(x) = exp(η(x...));
pde = D_xV(η(xSym...)) ~ 0.0f0; # placeholder pde

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
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));;

initθ = DiffEqFlux.initial_params(chain) #|> gpu;
eltypeθ = eltype(initθ)

parameterless_type_θ = DiffEqBase.parameterless_type(initθ);

strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = xSym
depvars = [η(xSym...)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

##
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
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);
train_domain_set = train_domain_set #|> gpu
train_domain_set_cpu = Array(train_domain_set[1])
nTrainDomainSet = size(train_domain_set[1],2)

function pde_loss_function_custom(θ)
    # custom self-written pde loss function

    # # TYPE 1
    # function ηNetS(x)
    #     return first(Array(phi(x, θ)))
    # end
    # ρNetS(x) = exp(ηNetS(x)) 
    # df(x) = ForwardDiff.jacobian(f, x)
    # dη(x) = ForwardDiff.gradient(ηNetS, x)
    # d2η(x) = ForwardDiff.jacobian(dη, x)
    
    # pdeErr(x) = (tr(df(x)) + dot(f(x), dη(x)) - Q_fpke / 2 * (sum(d2η(x)) + sum(dη(x)*dη(x)')))^2

    # # TYPE 2
    ρFn(y) = exp(first(Array(phi(y, θ)))); # phi is the NN representing η

    fxρ(y) = f(y)*ρFn(y);
    gxρ(y) = 0.5 * (g(y) * Q * g(y)') * ρFn(y);
    
    function term1(y)
        tmp = ForwardDiff.jacobian(fxρ, y)
        return sum(diag(tmp))
    end
    function term2(y) # takes too long to compute
        tmp = Q_fpke/2*sum(ForwardDiff.hessian(ρFn,y))
        return tmp
    end
    pdeErr(y) = (1/ρFn(y)*((-term1(y)) + term2(y)))^2; # pdeErr evaluated at state y
        
    # errCols = mapslices(pdeErr, train_domain_set_cpu, dims = 1); # pde error for each column/state in training set (slightly better than map)
    # errCols = map(pdeErr, eachslice(train_domain_set_cpu, dims = 2)); # pde error for each column/state in training set
    # tmp = Float32(sum(errCols)/nTrainDomainSet); # mean squared error

    tmp = (sum(pdeErr(train_domain_set_cpu[:,i]) for i in 1:nTrainDomainSet)/nTrainDomainSet) #mean squared error
    return Float32.(tmp)
end

@show pde_loss_function_custom(initθ)
##
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
term1_losses = Float32[];
term2_losses = Float32[];
BC_losses = Float32[];
cb_ = function (p, l)
    global nSteps = nSteps + 1
    println("[$nSteps] Current loss is: $l")
    println(
        "Individual losses are: PDE loss:",
        pde_loss_function_custom(p),
        "; BC loss:",
        bc_loss_function_sum(p),
    )

    push!(PDE_losses, pde_loss_function_custom(p))
    push!(BC_losses, bc_loss_function_sum(p))
    
    if runExp
        open("out_fullpde/log$(expNum).txt", "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam = Array(p), PDE_losses, BC_losses); ## save for checking.
    end
    return false
end

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt, cb = cb_, maxiters = maxOptIters);
println("Optimization done.");

## Save data
cd(@__DIR__);
if runExp
    jldsave(saveFile; optParam = Array(res.minimizer), PDE_losses, BC_losses);
end
