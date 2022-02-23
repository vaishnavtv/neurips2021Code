## Solve the FPKE for the 1d example using baseline PINNs (large training set) using custom written loss function (not using NeuralPDE to generate pde loss fn)
cd(@__DIR__);
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra
using ForwardDiff, Statistics
println("Packages loaded.")

import Random:seed!; seed!(1);

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2
Q_fpke = 0.25f0; # Q = σ^2

dx = 0.01; # discretization size used for training

expNum = 1;
runExp = false;
useGPU = true;
saveFile = "data_custom_1d/custom_ex1d_exp$(expNum).jld2";
runExp_fileName = "out_ad_self/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Steady State 1D with grid training. 2 HL with $(nn) neurons in the hl and $(string(activFunc)) activation. $(maxOpt1Iters) iterations with $(opt1) and then $(maxOpt2Iters) with $(opt2). Q_fpke = $(Q_fpke). useGPU = $(useGPU).
        Experiment number: $(expNum)\n")
    end
end

## set up the NeuralPDE framework using low-level API
@parameters x1
@variables  η(..)

xSym = x1;

# 1D Dynamics
α = 0.3f0; β = 0.5f0;
f(x) = (α*x - β*x.^3);
df(x) = Zygote.jacobian(f,x)[1];
# Zygote.jacobian(f, [0.0])[1]
# df([0.0])
# tx0 = train_domain_set[1][:,1:2]
# [df(tx0[:,i]) for i in 1:2]
# tx = diag(df(train_domain_set[1][:,1:2]))
g(x) = 1.0f0;

# PDE
ρ_true(x) = exp((1/(2*Q_fpke))*(2*α*x^2 - β*x^4)); # true analytical solution, before normalization
ρ(x) = exp(-η(xSym));
F = f(xSym)*ρ(xSym);
#  PDE written directly in η
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G = diffC*η(xSym...);

T1 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(η(xSym...)))) for i in 1:length(xSym)]); # drift term
T2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
T2 += sum([(Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])  - (Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])*η(xSym...) + diffC[i,j]*abs2(Differential(xSym[i])(η(xSym...))) for i in 1:length(xSym), j=1:length(xSym)]); # complete diffusion term, modified for GPU

Eqn = expand_derivatives(-T1+T2); 
pdeOrig = simplify(Eqn, expand = true) ~ 0.0f0;
pde = pdeOrig;
println("PDE defined.");

## Domain
maxval = 2.2f0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval)];

# Boundary conditions
bcs = [ρ(-maxval) ~ 0.0f0, ρ(maxval) ~ 0.0f0]

## Neural network
dim = 1 # number of dimensions
chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));
# chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));

initθ = DiffEqFlux.initial_params(chain);
if useGPU
    initθ = initθ |> gpu;
end
flat_initθ = initθ;
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = [x1]
depvars = [η(x1)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

train_domain_set, train_bound_set =
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);
if useGPU
    train_domain_set = train_domain_set |> gpu;
    train_bound_set = train_bound_set |> gpu;
end

_pde_loss_function = NeuralPDE.build_loss_function(
    pde,
    indvars,
    depvars,
    phi,
    derivative,
    integral,
    chain,
    initθ,
    strategy,
);
##
using ForwardDiff, FiniteDifferences
m = central_fdm(5,1);
g1(z) = Array(phi(z,initθ));
dg1_fD(z) = Zygote.jacobian(g1, [z])[1];
# dg1_finD(z) = FiniteDifferences.jacobian(m, g1, [z])[1]
# @show dg1_fD(0.0);
# @show dg1_finD(0.0);

##
t1Fn(z) = sum(phi(z, initθ));
dt1Fn(z) = Zygote.jacobian(t1Fn,z)[1];
d2t1Fn(z) = Zygote.hessian(t1Fn,z)#[1];
# tx0 = train_domain_set[1][:,1:2];
# @show dt1Fn(tx0) # this is fine, works on both gpu and cpu
# @show d2t1Fn(tx0[:,2]) # not working on gpu, works on cpu

u_ = (cord, θ, phi)->sum(phi(cord, θ));
# @show dphi1 =derivative(phi, u_, tx0[:,2], [0.0049215667], 1, initθ ); # same as dt1Fn(tx0[:,2])
# @show dphi2 =derivative(phi, u_, tx0[:,2], [[0.0049215667], [0.0049215667]],2, initθ ); # same as d2t1Fn(tx0[:,2])

## symbolic loss function 
_depvars, _indvars, dict_indvars, dict_depvars, dict_depvar_input = NeuralPDE.get_vars(indvars, depvars);
expr_loss_function = NeuralPDE.build_symbolic_loss_function(pde,indvars,depvars,#,dict_indvars,dict_depvars,
dict_depvar_input,phi,derivative,integral,chain,initθ,strategy);

##
# loss_b = similar(train_domain_set[1]); # output of build loss function
# loss_b_tmp = similar(loss_b[:,1]);
function _custom_pde_loss_function(y, θ)
    # equivalent to _pde_loss_function (obtained from build_loss_function)
    fd_dphi1(z) = (derivative(phi, u_, z, [0.0049215667f0], 1, initθ)); # gradient of η wrt input using finite difference
    fd_dphi2(z) = derivative(phi, u_, z, [[0.0049215667f0], [0.0049215667f0]],2, initθ); # hessian of η wrt input using finite difference

    function pdeErr(z)
        out = 0f0;
        # fd_t1 = fd_dphi1(z).*f(z) .+ df(z); # drift term
        # fd_t2 = Q_fpke/2f0*(fd_dphi1(z).^2 .+ fd_dphi2(z)); # diffusion term
        # return (sum(-fd_t1 .+ fd_t2))

        out -= fd_dphi1(z)*f(sum(z)) + df(z);
        out += Q_fpke/2f0*(fd_dphi1(z).^2 .+ fd_dphi2(z));
        # out = sum(fd_dphi1(z).*f(z) .+ df(z);
        #     .+ Q_fpke/2*(fd_dphi1(z).^2));
        # return out
    end

    # for i in 1:size(y,2)
    #     loss_b[:,i] = pdeErr(@view y[:,i]);
    # end
    # return [pdeErr(@view y[:,i]) for i in 1:size(y,2)]
    out2 = similar(y);#Vector{Float32}(undef, size(y,2));
    for i in 1:size(y,2)
        out2[:,i] = pdeErr(@view y[:,i]);
    end
    return out2
end
# txg = ForwardDiff.jacobian(x->_custom_pde_loss_function(train_domain_set[1][:,1:2], x), initθ);
# @show txg

# function _custom_pde_loss_function_finD(y, θ)
#     # equivalent to _pde_loss_function (obtained from build_loss_function)
#     ηFn(z) = (Array(phi(z,θ)));
#     # ρFn(y) = exp(sum(Array(phi(y, θ))));
#     dηFn(z) = FiniteDifferences.jacobian(m,ηFn, z)[1];

#     function pdeErr(z)
#         t1 = FiniteDifferences.jacobian(m,f,[z])[1] + f(z)*FiniteDifferences.jacobian(m,ηFn,[z])[1];
#         t2 = Q_fpke/2*((FiniteDifferences.jacobian(m,ηFn,[z])[1]).^2 + FiniteDifferences.jacobian(m,dηFn,[z])[1]);
#         return (-t1 + t2)
#     end

#     return pdeErr.(y)
# end
@info "Testing custom loss function...";
@show _pde_loss_function(train_domain_set[1][:,1:2], initθ)
@show _custom_pde_loss_function(train_domain_set[1][:,1:2], initθ);
# @show _custom_pde_loss_function_finD(train_domain_set[1][:,1:2], initθ);

#
using BenchmarkTools
# @btime _pde_loss_function(train_domain_set[1][:,1:5], initθ); #   1.146 ms (3691 allocations: 169.28 KiB)
# @btime _custom_pde_loss_function(train_domain_set[1][:,1:5], initθ); # 583.718 ms (1476030 allocations: 67.78 MiB) with NeuralPDE's derivative used for finite difference based gradients and hessians
# @btime _custom_pde_loss_function_finD(train_domain_set[1], initθ); # 1.166 s (7505843 allocations: 1.10 GiB) # the worst

##
# function pdeErr!(loss_b_tmp, z)
#     loss_b_tmp[1] = sum(-(fd_dphi1(z).*f(z) .+ df(z))); # drift term
#     loss_b_tmp[1] += sum(Q_fpke/2*(fd_dphi1(z).^2 .+ fd_dphi2(z))); # diffusion term
#     # loss_b_tmp[1] = sum(-fd_t1 + fd_t2)
#     # return sum(-fd_t1 .+ fd_t2)
# end
# @btime pdeErr(tmpY)
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


pde_loss_function = NeuralPDE.get_loss_function(
    _pde_loss_function,
    train_domain_set[1],
    eltypeθ,
    parameterless_type_θ,
    strategy,
);
@show pde_loss_function(initθ)

function custom_pde_loss_function(θ)
    # equivalent to pde_loss_function (obtained from NeuralPDE.get_loss_function)
    loss = mean(abs2, _custom_pde_loss_function(train_domain_set[1], θ));
end
@show custom_pde_loss_function(initθ)
##
bc_loss_functions = [
    NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy) for
    (loss, set) in zip(_bc_loss_functions, train_bound_set)
]

bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
@show bc_loss_function_sum(initθ)

nSteps = 0;
function loss_function_(θ, p)
    return custom_pde_loss_function(θ) + bc_loss_function_sum(θ) 
end
@show loss_function_(initθ,0)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoForwardDiff())
prob = GalacticOptim.OptimizationProblem(f_, initθ)

nSteps = 0;
PDE_losses = Float32[];
BC_losses = Float32[];
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
    )

    push!(PDE_losses, pde_loss_function(p))
    push!(BC_losses, bc_loss_function_sum(p))

    return false
end

println("Calling GalacticOptim()");
# res = GalacticOptim.solve(prob, opt1, cb=cb_, maxiters=maxOpt1Iters);
# prob = remake(prob, u0=res.minimizer)
# res = GalacticOptim.solve(prob, opt2, cb=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

if runExp
    jldsave(saveFile;optParam = Array(res.minimizer), PDE_losses, BC_losses);
end