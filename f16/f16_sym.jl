## Solve the FPKE for the nonlinear F16 dynamics with LQR controller using baseline PINNs (large training set)
cd(@__DIR__);
mkpath("out_sym")
mkpath("data_sym")
include("f16_controller.jl")
# include("NonlinearF16Model.jl")
# https://archive.siam.org/books/dc11/f16/Model.pdf for F16 Model Dynamics parameters

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, Symbolics, JLD2, ForwardDiff, Revise
using F16Model
   
import Random:seed!;
seed!(1);

# parameters for neural network
nn = 100; # number of neurons in the hidden layers
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

Q_fpke = 0.0f0; # Q = σ^2

expNum = 1;
saveFile = "data_sym/f16Sym_$(expNum).jld2";
useGPU = true;
runExp = false; # flag to check if running batch file
runExp_fileName = ("out_sym/log$(expNum).txt");
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "F16 SS after some symbolic manipulation. 2 HL with $(nn) neurons in the hl and $(activFunc) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS.  using GPU? $(useGPU). Q_fpke = $(Q_fpke).\n")
    end;
end
##
# Nominal Controller for Longitudinal F16Model trimmmed at specified altitude and velocity in 
# Trim vehicle at specified altitude and velocity
h0 = 10000f0; # ft
Vt0 = 500f0;   # ft/s
##
maskIndx = zeros(Float32,(length(xbar),4));
maskIndu = zeros(Float32,(length(ubar),2));
for i in 1:4
    maskIndx[ind_x[i],i] = 1f0;
    if i<3
        maskIndu[ind_u[i],i] = 1f0;
    end
end
##
function f16Model_4x(x4, xbar, ubar, Kc)
    # nonlinear dynamics of 4-state model with stabilizing controller
    # x4 are the states, not perturbations

    xFull = xbar2 + maskIndx*Array(x4);

    u = (Kc*(Array(x4) - xbar[ind_x]));
    uCont = Zygote.@ignore contSat(u)
    uFull = ubar2 + maskIndu*uCont

    xdotFull = F16Model.Dynamics(xFull, uFull)

    return (xdotFull[ind_x]) # return the 4 state dynamics
end

using IfElse
function contSat(u)
    # controller saturation
    uRet = similar(u)
    uRet .= u;
    uRet[2] = IfElse.ifelse(u[2]<-25.0f0, -25.0f0, u[2]);
    uRet[2] = IfElse.ifelse(u[2]>25.0f0, u[2], 25.0f0);

    return uRet
end

@parameters xV, xα, xθ, xq
@variables η(..)

xSym = [xV, xα, xθ, xq]

f(x) = f16Model_4x(x, xbar, ubar, Kc);
df(x) = ForwardDiff.jacobian(f,x);

f1(x1,x2,x3,x4) = f([x1,x2,x3,x4]);
df1(x1,x2,x3,x4) = df([x1,x2,x3,x4]);

# list of f terms as functions
fL1(x1,x2,x3,x4) = f([x1,x2,x3,x4])[1];
fL2(x1,x2,x3,x4) = f([x1,x2,x3,x4])[2];
fL3(x1,x2,x3,x4) = f([x1,x2,x3,x4])[3];
fL4(x1,x2,x3,x4) = f([x1,x2,x3,x4])[4];


# list of df diagonal terms as functions
dfL1(x1,x2,x3,x4) = df([x1,x2,x3,x4])[1,1];
dfL2(x1,x2,x3,x4) = df([x1,x2,x3,x4])[2,2];
dfL3(x1,x2,x3,x4) = df([x1,x2,x3,x4])[3,3];
dfL4(x1,x2,x3,x4) = df([x1,x2,x3,x4])[4,4];

# Registering functions
@register_symbolic f16Model_4x(x4, xbar, ubar, Kc)
@register_symbolic f1(x1,x2,x3,x4)
@register_symbolic df1(x1,x2,x3,x4)
@register_symbolic fL1(x1,x2,x3,x4)
@register_symbolic fL2(x1,x2,x3,x4)
@register_symbolic fL3(x1,x2,x3,x4)
@register_symbolic fL4(x1,x2,x3,x4)
@register_symbolic dfL1(x1,x2,x3,x4)
@register_symbolic dfL2(x1,x2,x3,x4)
@register_symbolic dfL3(x1,x2,x3,x4)
@register_symbolic dfL4(x1,x2,x3,x4)

## PDE
function g(x)
    return Float32.(1.0I(4)) # Float32.([1.0; 1.0; 1.0; 1.0]) # diffusion in all states(?)
end

D_xV = Differential(xV);
D_xα = Differential(xα);
D_xθ = Differential(xθ);
D_xq = Differential(xq);

ρ(x) = exp(η(x[1], x[2], x[3], x[4]));
# F = f1(xSym...)*ρ(xSym);
# # G = 0.5f0 * (g(xSym) * Q * g(xSym)') * ρ(xSym);
diffC = 0.5f0 * g(xSym) * Q_fpke * g(xSym)'; # diffusion coefficient D
G = diffC * ρ(xSym);

T1 = (dfL1(xSym...) + dfL2(xSym...) + dfL3(xSym...) + dfL4(xSym...)) + dot([fL1(xSym...), fL2(xSym...), fL3(xSym...), fL4(xSym...)],[Differential(xSym[i])(η(xSym...)) for i in 1:length(xSym)])
pdeOrig = simplify((T1), expand = true) ~ 0.0f0; # equation directly in η
pdeFn(xSym) = (dfL1(xSym...) + dfL2(xSym...) + dfL3(xSym...) + dfL4(xSym...));
# T2 = sum([(Differential(xSym[i]) * Differential(xSym[j]))(G[i, j]) for i = 1:length(xSym), j = 1:length(xSym) ]); # diffusion term to be expanded
# pdeOrig = simplify((T1 - T2)/ρ(xSym), expand = true) ~ 0.0f0;

pde = pdeOrig;

println("PDE defined")

## Domain
xV_min = 400f0; xV_max = 800f0;
xα_min = deg2rad(-15f0); xα_max = deg2rad(30f0);
xθ_min = xα_min; xθ_max = xα_max;
xq_min = -pi / 6f0; xq_max = pi / 6f0;
domains = [xV ∈ IntervalDomain(xV_min, xV_max), xα ∈ IntervalDomain(xα_min, xα_max), xθ ∈ IntervalDomain(xθ_min, xθ_max), xq ∈ IntervalDomain(xq_min, xq_max),];

## Grid discretization
dV = 10.0f0; dα = deg2rad(5f0); 
dθ = dα; dq = deg2rad(5f0);
dx = [dV; dα; dθ; dq]; # grid discretization in V (ft/s), α (rad), θ (rad), q (rad/s)


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
chain = Chain(Dense(dim, nn, activFunc), Dense(nn,nn,activFunc), Dense(nn, 1));;

initθ = DiffEqFlux.initial_params(chain) #|> gpu;
if useGPU
    using CUDA
    CUDA.allowscalar(false)
    initθ = initθ |> gpu;
end
th0 = initθ;
eltypeθ = eltype(initθ)

parameterless_type_θ = DiffEqBase.parameterless_type(initθ);

strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = xSym
depvars = [η(xSym...)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

##
train_domain_set, train_bound_set = NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);
if useGPU
    train_domain_set = train_domain_set |> gpu;
    train_bound_set = train_bound_set |> gpu;
end
nTrainDomainSet = size(train_domain_set[1], 2)


_pde_loss_function = NeuralPDE.build_loss_function(pde,indvars,depvars,phi,derivative,integral,chain,initθ,strategy);
tx = train_domain_set[1][:,1:5];
@show _pde_loss_function(tx, initθ)
##

bc_indvars = NeuralPDE.get_argument(bcs, indvars, depvars);
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,phi,derivative,integral,chain,initθ,strategy,bc_indvars=bc_indvar,) for (bc, bc_indvar) in zip(bcs, bc_indvars)];

pde_loss_function = NeuralPDE.get_loss_function(_pde_loss_function, train_domain_set[1],  eltypeθ, parameterless_type_θ, strategy);
@show pde_loss_function(initθ);

dphi(x,θ) = Zygote.gradient(th->first(phi(x,th)), θ)[1];
dphiX(x,θ) = Zygote.gradient(y->first(phi(y,θ)), x)[1];
d_pdeL(θ) = Zygote.gradient(pde_loss_function, θ)[1];
@show maximum(abs.(d_pdeL(th0)))
##
bc_loss_functions = [
    NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy) for (loss, set) in zip(_bc_loss_functions, train_bound_set)
]
bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
@show bc_loss_function_sum(initθ)

function loss_function_(θ, p)
    return pde_loss_function(θ) + bc_loss_function_sum(θ)
    # return combined_pde_loss_function(θ) + bc_loss_function_sum(θ)
end
@show loss_function_(initθ, 0)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, initθ)

nSteps = 0;
PDE_losses = Float32[];
losses = Float32[];
term2_losses = Float32[];
BC_losses = Float32[];
cb_ = function (p, l)
    global nSteps = nSteps + 1
    println("[$nSteps] Current loss is: $l")
    println(
        "Individual losses are: PDE loss:",
        pde_loss_function(p),
        "; BC loss:",
        bc_loss_function_sum(p),
    )

    push!(PDE_losses, pde_loss_function(p))
    push!(BC_losses, bc_loss_function_sum(p))

    if runExp
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses, BC_losses);
    end 
    return false
end

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt1, callback=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer);
res = GalacticOptim.solve(prob, opt2, callback=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

## Save data
cd(@__DIR__);
if runExp
    jldsave(saveFile; optParam=Array(res.minimizer), PDE_losses, BC_losses);
end

