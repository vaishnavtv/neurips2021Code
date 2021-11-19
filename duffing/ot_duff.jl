## Solve the FPKE for the Duffing oscillator using OT PINNs

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, Symbolics, JLD2, LinearAlgebra
using ForwardDiff, MosekTools, Convex
cd(@__DIR__);

import Random: seed!;
seed!(1);

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = Optim.BFGS();#ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.BFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 1000; # maximum number of training iterations for opt2
opt = Optim.BFGS(); # Optimizer used for training in OT
maxOptIters = 1000; # maximum number of training iterations
α_bc = 1.0f0;
otIters = 20; # number of OT iterations
maxNewPts = 200; # number of points added each OT iteration

## Grid discretization
dx = 0.05;

suff = string(activFunc);
runExp = true; 
expNum = 3;
saveFile = "data_ot/ot_duff_exp$(expNum).jld2";
runExp_fileName = "out_ot/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "SS Duffing Oscillator with OT and dx = $(dx). 2 HL with $(nn) neurons in the hl and $(suff) activation. Boundary loss coefficient: $(α_bc). Iteration 0 with 2 opts. $(maxOpt1Iters) iterations with BFGS and $(maxOpt2Iters) with BFGS. Then, running OT for $(otIters) iters, $(maxNewPts) each iter. opt: BFGS for $(maxOptIters). Diffusion in α. Q_fpke = 0.1f0. Using only unique new points.
        Experiment number: $(expNum)\n")
    end
end
# Duffing Oscillator Dynamics
η_duff = 0.2; α_duff = 1.0; β_duff = 0.2;
f(x) = [x[2]; η_duff.*x[2] .- α_duff.*x[1] .- β_duff.*x[1].^3]; # dynamics

function g(x::Vector)
    return [0.0f0;1.0f0];
end # diffusion


@parameters x1, x2
@variables η(..)

xSym = [x1; x2]

# PDE
ρ(x) = exp(η(x...));
Q_fpke = 0.01f0#*1.0I(2); # σ^2
#  PDE written directly in η
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G = diffC*η(xSym...);

T1 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(η(xSym...)))) for i in 1:length(xSym)]); # drift term
T2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
T2 += sum([(Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j]) - (Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])*η(xSym...) + diffC[i,j]*(Differential(xSym[i])(η(xSym...)))*(Differential(xSym[j])(η(xSym...))) for i in 1:length(xSym), j=1:length(xSym)]); # complete diffusion term

Eqn = expand_derivatives(-T1+T2); 
pdeOrig = simplify(Eqn, expand = true) ~ 0.0f0;
pde = pdeOrig;


## Domain
maxval = 2.0f0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval),
           x2 ∈ IntervalDomain(-maxval,maxval)];

# Boundary conditions
bcs = [ρ([-maxval,x2]) ~ 0.f0, ρ([maxval,x2]) ~ 0,
       ρ([x1,-maxval]) ~ 0.f0, ρ([x1,maxval]) ~ 0];


## Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1)) ;#|> gpu;

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
@show bc_loss_function_sum(initθ)

function loss_function_(θ, p)
    return pde_loss_function(θ) + α_bc*bc_loss_function_sum(θ)  
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
CsEval = collocationGrid(maxval * [-1, 1], maxval * [-1, 1], nEvalFine);

## initialize variables for OT-iterations
pde_train_sets = Vector{Vector{Matrix{Float32}}}(undef, otIters + 1);
pde_train_sets[1] = train_domain_set;
# pdeLossFunctions = Vector{Function}(undef, otIters + 1);
# pdeLossFunctions[1] = pde_loss_function;
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
            kron(ones(1, n), 1.0I(n))
            kron(1.0I(n), ones(1, n))
        ])

        ϕ = Convex.Variable(n^2, Positive())
        p = minimize(sum(C[:] .* ϕ), Aeq * ϕ == beq)
        # solve!(p,()->COSMO.Optimizer());
        Convex.solve!(p, () -> Mosek.Optimizer(LOG=0))

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
        y_un = unique(y, dims = 2); # only return unique new points

        return y_un
    end
    ##

    y = newPtsFn(CsEval, maxNewPts) # new points generated by OT
    CsNew = [Cs y] # add to training set
    newPtsAll[i] = y
    pde_train_sets[i+1] = [CsNew]

    # remake pde loss function
    pde_loss_function = NeuralPDE.get_loss_function(
        _pde_loss_function,
        pde_train_sets[i+1][1],
        eltypeθ,
        parameterless_type_θ,
        strategy,
    );

    # remake optimization loss function
    function loss_function_i(θ, p)
        return pde_loss_function(θ) + α_bc*bc_loss_function_sum(θ)
    end
    @show loss_function_i(initθ, 0)

    global nSteps = 0
    PDE_losses[i+1] = Float32[]
    BC_losses[i+1] = Float32[]
    cb_ = function (p, l)
        global nSteps = nSteps + 1
        println("[$nSteps] Current loss is: $l")
        println(
            "Individual losses are: PDE loss:",
            pde_loss_function(p),
            ", BC loss:",
            bc_loss_function_sum(p),
        )

        push!(PDE_losses[i+1], pde_loss_function(p))
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