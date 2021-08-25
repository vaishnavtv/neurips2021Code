## Solve the FPKE for the Van der Pol Rayleigh oscillator using OT-PINNs
include("../otSampler_RB/rbfNet.jl")

# import prerequisite packages
using NeuralPDE,
    Flux,
    ModelingToolkit,
    GalacticOptim,
    Optim,
    DiffEqFlux,
    Symbolics,
    JLD2,
    Convex,
    MosekTools,
    ForwardDiff,
    LinearAlgebra
using CUDA
CUDA.allowscalar(false)

import Random: seed!;
seed!(1);

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
maxOptIters = 10000; # maximum number of training iterations
otIters = 20; # maximum number of OT iterations
maxNewPts = 200; # maximum new points found through OT in each iteration

Q = 0.3; # Q = σ^2

dx = 0.25; # discretization size for generating data for the nominal network

opt = Optim.LBFGS(); # optimizer used for training
# opt = ADAM(1e-3); 

suff = string(activFunc);
saveFileLoc = "data/dx25eM2_ot1Eval_vdpr_$(suff)_$(nn)_ot$(otIters)_mnp$(maxNewPts)_gpu_otShStab.jld2";

## set up the NeuralPDE framework using low-level API
@parameters x1, x2
@variables η(..)

x = [x1; x2]

# Van der Pol Rayleigh Dynamics
f(x) = [x[2]; -x[1] + (1 - x[1]^2 - x[2]^2) * x[2]];

function g(x)
    return [0.0; 1.0]
end

# PDE
ρ(x) = exp(η(x[1], x[2]));
F = f(x) * ρ(x);
G = 0.5 * (g(x) * Q * g(x)') * ρ(x);

T1 = sum([Differential(x[i])(F[i]) for i = 1:length(x)]);
T2 = sum([
    (Differential(x[i]) * Differential(x[j]))(G[i, j]) for i = 1:length(x), j = 1:length(x)
]);

Eqn = expand_derivatives(-T1 + T2); # + dx*u(x1,x2)-1 ~ 0;
pdeOrig = simplify(Eqn / ρ(x)) ~ 0;
pde = (0.15f0Differential(x2)(Differential(x2)(η(x1, x2)))*exp(η(x1, x2)) + 0.15f0exp(η(x1, x2))*(Differential(x2)(η(x1, x2))^2) - (exp(η(x1, x2))*(1 - (x1^2) - (3(x2^2)))) - (x2*Differential(x1)(η(x1, x2))*exp(η(x1, x2))) - (Differential(x2)(η(x1, x2))*exp(η(x1, x2))*(x2*(1 - (x1^2) - (x2^2)) - x1)))*(exp(η(x1, x2))^-1) ~ 0.0f0; # simplified pde rewritten with constants in float32 format

# Domain
maxval = 2.0;
domains = [x1 ∈ IntervalDomain(-maxval, maxval), x2 ∈ IntervalDomain(-maxval, maxval)];

# Boundary conditions
bcs = [
    ρ([-maxval, x2]) ~ 0.0f0,
    ρ([maxval, x2]) ~ 0,
    ρ([x1, -maxval]) ~ 0.0f0,
    ρ([x1, maxval]) ~ 0,
];

## Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));

initθ = DiffEqFlux.initial_params(chain) |> gpu;
flat_initθ = if (typeof(chain) <: AbstractVector)
    reduce(vcat, initθ)
else
    initθ
end
eltypeθ = eltype(flat_initθ)

parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);
strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = [x1, x2]
depvars = [η]

_pde_loss_function = NeuralPDE.build_loss_function(
    pde,
    indvars,
    depvars,
    phi,
    derivative,
    chain,
    initθ,
    strategy,
);

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
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);
train_domain_set = train_domain_set |> gpu

pde_loss_function = NeuralPDE.get_loss_function(
    _pde_loss_function,
    train_domain_set[1],
    eltypeθ,
    parameterless_type_θ,
    strategy,
);
@show pde_loss_function(initθ)

bc_loss_functions = [
    NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy) for
    (loss, set) in zip(_bc_loss_functions, train_bound_set)
]

bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
@show bc_loss_function_sum(initθ)

function loss_function_(θ, p)
    return pde_loss_function(θ) + bc_loss_function_sum(θ)
end
@show loss_function_(initθ,0)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, initθ)

nSteps = 0;
PDE_losses1 = Float32[];
BC_losses1 = Float32[];
cb_ = function (p, l)
    global nSteps = nSteps + 1
    println("[$nSteps] Current loss is: $l")
    println(
        "Individual losses are: PDE loss:",
        pde_loss_function(p),
        ", BC loss:",
        bc_loss_function_sum(p),
    )

    push!(PDE_losses1, pde_loss_function(p))
    push!(BC_losses1, bc_loss_function_sum(p))
    return false
end


println("Calling GalacticOptim()");

res = GalacticOptim.solve(prob, opt, cb = cb_, maxiters = maxOptIters);
println("Optimization 0 done."); # Network trained with nominal data

optParam1 = res.minimizer; # Weights discovered after nominal training

println("Starting OT iterations:");
## evaluation grid
nEvalFine = 100;
function collocationGrid(x, y, N)
    X = range(x[1], x[2], length = N)
    Y = range(y[1], y[2], length = N)
    XX = zeros(N, N)
    YY = similar(XX)
    for i = 1:N, j = 1:N
        XX[i, j] = X[i]
        YY[i, j] = Y[j]
    end
    Cs = [XX[:] YY[:]]'
    return Cs
end
CsEval = collocationGrid(maxval * [-1, 1], maxval * [-1, 1], nEvalFine);

## initialize variables for OT-iterations
pde_train_sets = Vector{typeof(train_domain_set)}(undef, otIters + 1);
pde_train_sets[1] = train_domain_set;
pdeLossFunctions = Vector{Function}(undef, otIters + 1);
pdeLossFunctions[1] = pde_loss_function;
newPtsAll = Vector{typeof(train_domain_set[1])}(undef, otIters);
optParams = Vector{typeof(optParam1)}(undef, otIters + 1);
optParams[1] = optParam1;
PDE_losses = Vector{Vector{Float32}}(undef, otIters + 1);
PDE_losses[1] = PDE_losses1;
BC_losses = Vector{Vector{Float32}}(undef, otIters + 1);
BC_losses[1] = BC_losses1;

## Run the OT-PINNs loop
for i = 1:otIters
## generate functions using new weights, biases
CUDA.allowscalar(true) # have to do this for OT (probably not recommended, investigate alternatives)
    function ρ_pdeErr_fns(optParam)
        # optParamArr = Array(optParam)
        function ηNetS(x)
            return first(phi(x, optParam))
        end
        ρNetS(x) = exp(ηNetS(x)) # solution after first iteration
        df(x) = ForwardDiff.jacobian(f, x)
        dη(x) = ForwardDiff.gradient(ηNetS, x)
        d2η(x) = ForwardDiff.jacobian(dη, x)

        pdeErrFn(x) = tr(df(x)) + dot(f(x), dη(x)) - Q / 2 * (d2η(x)[end] + (dη(x)[end])^2)
        return ρNetS, pdeErrFn
    end
    ρFn, pdeErrFn = ρ_pdeErr_fns(optParams[i])
    Cs = first(pde_train_sets[i]) # collocation points used by optimizer

    ## OT
    function otMapSp(X, W1, W2) # X is dxN matrix, d in dimension, N is number of samples. W1, W2 are column vectors
        # W1 : vector of size N (uniform value = 1/N)
        # W2: error distribution
        println("Solving OT problem")

        n = size(X, 2)

        C = zeros(n, n)
        for i = 1:n
            for j = 1:n
                dx = X[:, i] - X[:, j]
                C[i, j] = norm(dx)^2
            end
        end

        beq = [W2; W1]
        # Creation of Aeq takes 50% of time.
        Aeq = ([
            kron(ones(1, n), I(n))
            kron(I(n), ones(1, n))
        ])

        ϕ = Convex.Variable(n^2, Positive())
        p = minimize(sum(C[:] .* ϕ), Aeq * ϕ == beq)
        # solve!(p,()->COSMO.Optimizer());
        Convex.solve!(p, () -> Mosek.Optimizer())

        xx = ϕ.value

        W = sqrt(p.optval)
        D = maximum(C[:]) # Support diameter
        Phi = reshape(xx, n, n)
        return W, D, Phi
    end

    ## Take top maxNewPts corresponding to highest eqnErrColl;
    function newPtsFn(Cs, maxNewPts)
        nNN = size(Cs, 2)
        eqnErrCs = [(pdeErrFn(Cs[:, i]))^2 for i = 1:nNN] # equation error on collocation points Cs
        indSort = sortperm(eqnErrCs[:], rev = true) # sort in descending order
        indMaxErr = indSort[1:maxNewPts] # indices corresponding to maximum error (first maxNewPts)

        Cs_ot = Cs[:, indMaxErr]
        w1 = ones(maxNewPts) / maxNewPts # uniform distribution
        w2 = eqnErrCs[indMaxErr]
        w2 = w2 / sum(w2) # error distribution

        # W, D, Phi_sp = otMapSp(Cs_ot,w1,w2);
        # y = Cs_ot * (maxNewPts * Phi_sp)

        ## trying sinkhorn using new code
        otOpt = Mosek.Optimizer(LOG = 0) # Fast
        Phi_ot =
            otMap_gpu(Cs_ot, w1, w2, otOpt, alg = :sinkhorn_stab, maxIter = 50000, α = 0.001)

        y = Array(Cs_ot) * (maxNewPts * Array(Phi_ot')) # need transpose on Phi if using OptimalTransport package

        return cu(y)
    end


    y = newPtsFn(CsEval, 200) # new points generated by OT
    ##
    CsNew = [Cs y] # add to training set
    newPtsAll[i] = y
    pde_train_sets[i+1] = [CsNew]

    # remake pde loss function
    pdeLossFunctions[i+1] = NeuralPDE.get_loss_function(
        _pde_loss_function,
        pde_train_sets[i+1][1],
        eltypeθ,
        parameterless_type_θ,
        strategy,
    )

    # remake optimization loss function
    function loss_function_i(θ, p)
        return pdeLossFunctions[i+1](θ) + bc_loss_function_sum(θ)
    end

    global nSteps = 0
    PDE_losses[i+1] = Float32[]
    BC_losses[i+1] = Float32[]
    local cb_ = function (p, l)
        global nSteps = nSteps + 1
        println("[$nSteps] Current loss is: $l")
        println(
            "Individual losses are: PDE loss:",
            pdeLossFunctions[i+1](p),
            ", BC loss:",
            bc_loss_function_sum(p),
        )

        push!(PDE_losses[i+1], pdeLossFunctions[i+1](p))
        push!(BC_losses[i+1], bc_loss_function_sum(p))
        return false
    end

    # set up GalacticOptim optimization problem
    local f_ = OptimizationFunction(loss_function_i, GalacticOptim.AutoZygote())
    local prob = GalacticOptim.OptimizationProblem(f_, optParams[i])

    println("Calling GalacticOptim() after $(i) iterations of OT sampling")
    local res = GalacticOptim.solve(prob, opt, cb = cb_, maxiters = maxOptIters)
    println("Optimization done.")

    optParams[i+1] = res.minimizer
end
## convert to CPU array
pde_train_sets_cpu = Vector{Vector{Matrix{Float32}}}(undef, otIters + 1) 
for i in 1:otIters + 1
    pde_train_sets_cpu[i] = Array.(pde_train_sets[i]);
end
@show typeof(pde_train_sets_cpu)
## save data
cd(@__DIR__);
jldsave(saveFileLoc; optParams = Array.(optParams), PDE_losses, BC_losses, pde_train_sets = pde_train_sets_cpu, newPtsAll = Array.(newPtsAll));
println("Data saved.");
