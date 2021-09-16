## Solve the FPKE for the missile using baseline PINNs (large training set)

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, Symbolics, JLD2
using ForwardDiff, MosekTools, Convex
cd(@__DIR__);
include("missileDynamics.jl");

import Random: seed!;
seed!(1);

## parameters for neural network
nn = 20; # number of neurons in the hidden layer
activFunc = tanh; # activation function
maxOptIters = 1000; # maximum number of training iterations
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 2000; # maximum number of training iterations for opt1
opt2 = Optim.BFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 1000; # maximum number of training iterations for opt2
opt = Optim.LBFGS(); # Optimizer used for training in OT
α_bc = 1.0;
otIters = 20;
maxNewPts = 200;
## Grid discretization
dM = 0.01; dα = 0.01;
dx = [dM; dα] # grid discretization in M, α (rad)

suff = string(activFunc);
runExp = true; 
expNum = 2;
saveFile = "dataOT/ll_grid_missile_$(suff)_$(nn)_exp$(expNum).jld2";
runExp_fileName = "outOT/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Missile with GridTraining and dx = $(dx). 2 HL with $(nn) neurons in the hl and $(suff) activation. Boundary loss coefficient: $(α_bc). Iteration 0 with 2 opts. $(maxOpt1Iters) iterations with ADAM and $(maxOpt2Iters) with BFGS. Then, running OT for $(otIters) iters, $(maxNewPts) each iter. opt: LBFGS for $(maxOptIters).
        Experiment number: $(expNum)\n")
    end
end
# Van der Pol Rayleigh Dynamics
@parameters x1, x2
@variables η(..)

xSym = [x1; x2]

# PDE
ρ(x) = exp(η(x...));
Q_fpke = 0.1f0*1.0I(2); # σ^2
F = f(xSym) * ρ(xSym); # drift term
diffC = 0.5 * (g(xSym) * Q_fpke * g(xSym)'); # diffusion coefficient
G = diffC * ρ(xSym); # diffusion term

T1 = sum([Symbolics.derivative(F[i], xSym[i]) for i = 1:length(xSym)]); # pde drift term
T2 = sum([
    Symbolics.derivative(Symbolics.derivative(G[i, j], xSym[i]), xSym[j]) for i = 1:length(xSym), j = 1:length(xSym)
]); # pde diffusion term


Eqn = expand_derivatives(-T1 + T2); # + dx*u(x1,x2)-1 ~ 0;
pdeOrig = simplify(Eqn / ρ(xSym)) ~ 0.0f0;
pde = pdeOrig;

## Domain
minM = 1.2; maxM = 2.5;
minα = -1.0; maxα = 1.5;

domains = [x1 ∈ IntervalDomain(minM, maxM), x2 ∈ IntervalDomain(minα, maxα)];

# Boundary conditions
bcs = [
    ρ([minM, x2]) ~ 0.0f0,
    ρ([maxM, x2]) ~ 0.0f0,
    ρ([x1, minα]) ~ 0.0f0,
    ρ([x1, maxα]) ~ 0.0f0,
];

## Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1)) ;#|> gpu;

initθ = DiffEqFlux.initial_params(chain) #|> gpu;
flat_initθ = initθ
eltypeθ = eltype(flat_initθ)

parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = xSym
depvars = [η(xSym...)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

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
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);# |> gpu;
train_domain_set = train_domain_set #|> gpu
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Size of training dataset: $(size(train_domain_set[1],2))\n")
    end
end;

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
typeof(bc_loss_function_sum(initθ))
function loss_function_(θ, p)
    return pde_loss_function(θ) + α_bc*bc_loss_function_sum(θ)  
end
@show bc_loss_function_sum(initθ)
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

    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses1, BC_losses1);
    end
    return false
end

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt1, cb=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer)
res = GalacticOptim.solve(prob, opt2, cb=cb_, maxiters=maxOpt2Iters);
println("Optimization 0 done.");
optParam1 = res.minimizer;

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
CsEval = collocationGrid([minM, maxM], [minα, maxα] , nEvalFine);

## initialize variables for OT-iterations
pde_train_sets = Vector{Vector{Matrix{Float32}}}(undef, otIters + 1);
pde_train_sets[1] = train_domain_set;
pdeLossFunctions = Vector{Function}(undef, otIters + 1);
pdeLossFunctions[1] = pde_loss_function;
newPtsAll = Vector{Matrix{Float32}}(undef, otIters);
optParams = Vector{Vector{Float32}}(undef, otIters + 1);
optParams[1] = optParam1;
PDE_losses = Vector{Vector{Float32}}(undef, otIters + 1);
PDE_losses[1] = PDE_losses1;
BC_losses = Vector{Vector{Float32}}(undef, otIters + 1);
BC_losses[1] = BC_losses1;

## Run the OT-PINNs loop
for i = 1:otIters

    ## generate functions using new weights, biases
    function ρ_pdeErr_fns(optParam)
        function ηNetS(x)
            return first(phi(x, optParam))
        end
        ρNetS(x) = exp(ηNetS(x)) # solution after first iteration
        df(x) = ForwardDiff.jacobian(f, x)
        dη(x) = ForwardDiff.gradient(ηNetS, x)
        d2η(x) = ForwardDiff.jacobian(dη, x)

        pdeErrFn(x) = (tr(df(x)) + dot(f(x), dη(x)) - 1/2*sum(Q_fpke.*(d2η(x) + dη(x)*dη(x)')))
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
        w1 = ones(maxNewPts) / maxNewPts
        w2 = eqnErrCs[indMaxErr]
        w2 = w2 / sum(w2)

        _, _, Phi_sp = otMapSp(Cs_ot,w1,w2);
        y = Cs_ot*(maxNewPts*Phi_sp);

        return y
    end
    ##

    y = newPtsFn(CsEval, maxNewPts) # new points generated by OT
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
    cb_ = function (p, l)
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

        if runExp
            open(runExp_fileName, "a+") do io
                write(io, "otIter[$i]; [$(nSteps)] Current loss is: $l \n")
            end;
            
            jldsave(saveFile; optParams, PDE_losses, BC_losses, pde_train_sets, newPtsAll);
        end
        return false
    end

    # set up GalacticOptim optimization problem
    f_ = OptimizationFunction(loss_function_i, GalacticOptim.AutoZygote())
    prob = GalacticOptim.OptimizationProblem(f_, optParams[i])

    println("Calling GalacticOptim() after $(i) iterations of OT sampling")
    res = GalacticOptim.solve(prob, opt, cb = cb_, maxiters = maxOptIters)
    println("Optimization done.")

    optParams[i+1] = res.minimizer
end
## save data
if runExp
    jldsave(saveFile; optParams, PDE_losses, BC_losses, pde_train_sets, newPtsAll);
end
println("Data saved.");