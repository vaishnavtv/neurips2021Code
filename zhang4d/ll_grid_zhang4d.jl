# Dynamic system 5 from Kumar's PUFEM paper using quasi-strategy
# system taken from Zhang, Hao, et al. "Solving Fokker–Planck equations using deep KD-tree with a small amount of data." Nonlinear Dynamics (2022): 1-15.

using NeuralPDE, Flux, ModelingToolkit, Optimization, Optim, Symbolics, JLD2, DiffEqFlux
cd(@__DIR__);
mkpath("out_grid")
mkpath("data_grid")

import Random:seed!; seed!(1);
# using QuasiMonteCarlo

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 40000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

# file location to save data
dx = 0.25f0;

suff = string(activFunc);
expNum = 2;
useGPU = true;
saveFile = "data_grid/ll_grid_zhang4d_exp$(expNum).jld2";
runExp = true;
runExp_fileName = "out_grid/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Steady State 4D dynamics from Zhang's 2022 paper with Grid training. 2 HL with $(nn) neurons in the hl and $(suff) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS.  UniformSample strategy. PDE written directly in η. dx = $(dx). Using GPU? $(useGPU). PDE written manually in η. dx changed. Chagned maxval to 4.0f0.
        Experiment number: $(expNum)\n")
    end
end
## set up the NeuralPDE framework using low-level API
@parameters x1, x2, x3, x4
@variables  η(..)

xSym = [x1;x2; x3; x4]

## 4D Dynamics
a = 0.5f0; b = 1f0; k1 = -0.5f0; k2 = k1;
ϵ = 0.5f0; λ1 = 0.25f0; λ2 = 0.125f0; μ = 0.375f0;
M = 1f0; varI = 1f0;

vFn(x1,x2) = k1*x1^2 + k2*x2^2 + ϵ*(λ1*x1^4 + λ2*x2^4 + μ*x1^2*x2^2)
dvdx1_expr = Symbolics.derivative(vFn(x1,x2), x1);
dvdx1_fn(y1,y2) = substitute(dvdx1_expr, Dict([x1=>y1, x2=>y2]));
dvdx2_expr = Symbolics.derivative(vFn(x1,x2), x2);
dvdx2_fn(y1,y2) = substitute(dvdx2_expr, Dict([x1=>y1, x2=>y2]));

function f(x)
    output = [x[3]; x[4];
             -a*x[3] - 1/M*dvdx1_fn(x[1],x[2])
             -b*x[4] - 1/varI*dvdx2_fn(x[1],x[2])]
    return output
end

function g(x::Vector)
    return [0.0f0 0.0f0;0.0f0 0.0f0; 1.0f0 0.0f0; 0.0f0 1.0f0];
end

# PDE
println("Defining PDE");
Q_fpke = [2f0 0f0; 0f0 4f0;]; # Q = σ^2
ρ(x) = exp(η(xSym...));
# F = f(xSym)*ρ(xSym);
G = 0.5f0*(g(xSym)*Q_fpke*g(xSym)')*ρ(xSym);

# T1 = sum([Differential(xSym[i])(F[i]) for i in 1:length(xSym)]);
T2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);

# Eqn = expand_derivatives(-T1+T2); # + dx*u(x1,x2)-1 ~ 0;
# pdeOrig = simplify(Eqn/ρ(xSym)) ~ 0.0f0;
# pde = pdeOrig;
# println("PDE defined symbolically.")

# Equation written directly in η
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G2 = diffC*η(xSym...);

T1_2 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(η(xSym...)))) for i in 1:length(xSym)]); # drift term
# T2_2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G2[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
# T2_2 += sum([(Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j]) - (Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])*η(xSym...) + diffC[i,j]*(Differential(xSym[i])(η(xSym...)))*(Differential(xSym[j])(η(xSym...))) for i in 1:length(xSym), j=1:length(xSym)]); # complete diffusion term
T2_2 = 2.0f0*abs2(Differential(x4)(η(x1, x2, x3, x4))) + 2.0f0*Differential(x4)(Differential(x4)(η(x1, x2, x3, x4))) + abs2(Differential(x3)(η(x1, x2, x3, x4))) + Differential(x3)(Differential(x3)(η(x1, x2, x3, x4)))

Eqn = expand_derivatives(-T1_2+T2_2); 
pdeOrig2 = simplify(Eqn, expand = true) ~ 0.0f0;
pde = pdeOrig2;
println("PDE in  η defined symbolically.")

## Domain
maxval = 4.0f0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval),
           x2 ∈ IntervalDomain(-maxval,maxval),
           x3 ∈ IntervalDomain(-maxval,maxval),
           x4 ∈ IntervalDomain(-maxval,maxval)];

# Boundary conditions
bcs = [ρ([-maxval,x2,x3,x4]) ~ 0.f0, ρ([maxval,x2,x3,x4]) ~ 0,
       ρ([x1,-maxval,x3,x4]) ~ 0.f0, ρ([x1,maxval,x3,x4]) ~ 0,
       ρ([x1,x2,-maxval,x4]) ~ 0.f0, ρ([x1,x2,maxval,x4]) ~ 0,
       ρ([x1,x2,x3,-maxval]) ~ 0.f0, ρ([x1,x2,x3,maxval]) ~ 0,];

## Neural network
dim = 4 # number of dimensions
chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));

initθ = DiffEqFlux.initial_params(chain) #|> gpu;
if useGPU
    using CUDA
    CUDA.allowscalar(false)
    initθ = initθ |> gpu;
end
flat_initθ = initθ
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = xSym
depvars = [η(xSym...)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

_pde_loss_function = NeuralPDE.build_loss_function(pde,indvars,depvars,phi,derivative,integral,chain,initθ,strategy,);

bc_indvars = NeuralPDE.get_variables(bcs, indvars, depvars);
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,phi,derivative,integral,chain,initθ,strategy,bc_indvars = bc_indvar,) for (bc, bc_indvar) in zip(bcs, bc_indvars)]

train_domain_set, train_bound_set =
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars) ;
if useGPU
    train_domain_set = train_domain_set |> gpu;
    train_bound_set = train_bound_set |> gpu;
end

pde_loss_function = NeuralPDE.get_loss_function(_pde_loss_function,train_domain_set[1],eltypeθ,parameterless_type_θ,strategy,);
@show pde_loss_function(initθ)

bc_loss_functions = [
    NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy) for
    (loss, set) in zip(_bc_loss_functions, train_bound_set)
]

bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
@show bc_loss_function_sum(initθ)


nSteps = 0;
function loss_function_(θ, p)
    return pde_loss_function(θ) + bc_loss_function_sum(θ) 
end
@show loss_function_(initθ,0)

## set up Optimization optimization problem
f_ = OptimizationFunction(loss_function_, Optimization.AutoZygote())
prob = Optimization.OptimizationProblem(f_, initθ)

PDE_losses = Float32[];
BC_losses = Float32[];
cb_ = function (p, l)
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
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses, BC_losses);
    end
    return false
end

println("Calling Optimization()");
res = Optimization.solve(prob, opt1, callback=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer)
res = Optimization.solve(prob, opt2, callback=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

## Save data
cd(@__DIR__);
if runExp
    jldsave(saveFile;optParam = Array(res.minimizer), PDE_losses, BC_losses);
end