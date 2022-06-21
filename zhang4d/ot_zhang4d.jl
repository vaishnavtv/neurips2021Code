# Dynamic system 5 from Kumar's PUFEM paper using quasi-strategy
# system taken from Zhang, Hao, et al. "Solving Fokker–Planck equations using deep KD-tree with a small amount of data." Nonlinear Dynamics (2022): 1-15.

using NeuralPDE, Flux, ModelingToolkit, Optimization, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra, Convex, MosekTools
cd(@__DIR__);
mkpath("data_ot")
mkpath("out_ot")

import Random:seed!; seed!(1);
# using QuasiMonteCarlo

## parameters for neural network
nn = 48; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 1000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

## For OT
nOTIters = 20;
maxNewPts = 500;
dxFine = 0.25f0;

dx = 0.5f0;
# file location to save data
expNum = 4;
useGPU = true;
saveFile = "data_ot/ot_zhang4d_exp$(expNum).jld2";
runExp = true;
runExp_fileName = "out_ot/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "OT: Steady State 4D dynamics from Zhang's 2022 paper with Grid training. 2 HL with $(nn) neurons in the hl and $(activFunc) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS.  PDE written directly in η. dx = $(dx). Using GPU? $(useGPU). PDE written manually in η. 
        nOTIters = $(nOTIters). maxNewPts = $(maxNewPts). dxFine = $(dxFine).
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
ρ(xSym) = exp(η(xSym...));

# Equation written directly in η
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G2 = diffC*η(xSym...);

T1_2 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(η(xSym...)))) for i in 1:length(xSym)]); # drift term
T2_2 = 2.0f0*abs2(Differential(x4)(η(x1, x2, x3, x4))) + 2.0f0*Differential(x4)(Differential(x4)(η(x1, x2, x3, x4))) + abs2(Differential(x3)(η(x1, x2, x3, x4))) + Differential(x3)(Differential(x3)(η(x1, x2, x3, x4))) # diffusion term written manually

Eqn = expand_derivatives(-T1_2+T2_2); 
pdeOrig2 = simplify(Eqn, expand = true) ~ 0.0f0;
pde = pdeOrig2;
println("PDE in  η defined symbolically.")

## Domain
maxval = 5.0f0;
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

println("Calling Optimization()");
res = Optimization.solve(prob, opt1, callback=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer)
res = Optimization.solve(prob, opt2, callback=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

## Save data
cd(@__DIR__);
if runExp
    jldsave(saveFile;optParam = Array(res.minimizer), PDE_losses = [PDE_losses1], BC_losses = [BC_losses1]);
end

optParam1 = res.minimizer;

## OT ITERATIONS BEGIN
CsEvalSet, _ = NeuralPDE.generate_training_sets(domains, dxFine, [pde], bcs, eltypeθ, indvars, depvars) ;
CsEval = CsEvalSet[1] 
if useGPU
    CsEval = CsEvalSet[1] |> gpu;
end

## initialize variables for OT-iterations
pde_train_sets = Vector{typeof(train_domain_set)}(undef, nOTIters + 1);
pde_train_sets[1] = train_domain_set;
newPtsAll = Vector{typeof(train_domain_set[1])}(undef, nOTIters);
optParams = Vector{typeof(optParam1)}(undef, nOTIters + 1);
optParams[1] = optParam1;
PDE_losses = Vector{Vector{Float32}}(undef, nOTIters + 1);
PDE_losses[1] = PDE_losses1;
BC_losses = Vector{Vector{Float32}}(undef, nOTIters + 1);
BC_losses[1] = BC_losses1;

function otMapSp(X, W1, W2) # X is dxN matrix, d in dimension, N is number of samples. W1, W2 are column vectors
    # W1 : vector of size N (uniform value = 1/N)
    # W2: error distribution
    println("Solving OT problem")

    N = size(X, 2)

    C = zeros(N, N)
    for i = 1:N
        for j = 1:N
            dx = X[:, i] - X[:, j]
            C[i, j] = norm(dx)^2
        end
    end

    beq = [W2; W1]
    # Creation of Aeq takes 50% of time.
    Aeq = ([
        kron(ones(1, N), I(N))
        kron(I(N), ones(1, N))
    ])

    ϕ = Convex.Variable(N^2, Positive())
    p = minimize(sum(C[:] .* ϕ), Aeq * ϕ == beq)
    # solve!(p,()->COSMO.Optimizer());
    Convex.solve!(p, () -> Mosek.Optimizer())

    xx = ϕ.value

    W = sqrt(p.optval)
    D = maximum(C[:]) # Support diameter
    Phi = reshape(xx, N, N)
    return W, D, Phi
end

function newPtsFn(Cs, maxNewPts, optParam)
    # nNN = size(Cs, 2)
    eqnErrCs = abs2.(_pde_loss_function(Cs, optParam)) # equation error on collocation points Cs
    indSort = sortperm(eqnErrCs[:], rev = true) # sort in descending order
    indMaxErr = indSort[1:maxNewPts] # indices corresponding to maximum error (first maxNewPts)

    Cs_ot = Array(Cs[:, indMaxErr])
    w1 = ones(maxNewPts) / maxNewPts
    w2 = Array(eqnErrCs[indMaxErr])
    w2 = w2 / sum(w2)

    W, D, Phi_sp = otMapSp(Cs_ot,w1,w2);
    y = Cs_ot*(maxNewPts*Phi_sp);

    return (y)
end



for i=1:nOTIters
    ρFn(x) = exp(sum(phi(x, optParams[i])))
    # _pdeErrFn = _pdeLossFunctions[i];
    Cs = pde_train_sets[i][1];

    y = newPtsFn(CsEval, maxNewPts, optParams[i]) # new points generated by OT
    CsNew = [Cs y] # add to training set
    newPtsAll[i] = y
    pde_train_sets[i+1] = [CsNew]

    # remake pde loss function
    pdeLossFunction= NeuralPDE.get_loss_function(_pde_loss_function,pde_train_sets[i+1][1],eltypeθ,parameterless_type_θ,strategy)
    @show pdeLossFunction(optParams[i])

    # remake optimization loss function
    function loss_function(θ, p)
        return pdeLossFunction(θ) + bc_loss_function_sum(θ)
    end
    @show loss_function(optParams[i],0)

    global nSteps = 0
    PDE_losses[i+1] = Float32[]
    BC_losses[i+1] = Float32[]
    local cb_ = function (p, l)
        global nSteps = nSteps + 1
        println("[$nSteps] Current loss is: $l")
        println(
            "Individual losses are: PDE loss:",
            pdeLossFunction(p),
            ", BC loss:",
            bc_loss_function_sum(p),
        )

        push!(PDE_losses[i+1], pdeLossFunction(p))
        push!(BC_losses[i+1], bc_loss_function_sum(p))

        if runExp # if running job file
            open(runExp_fileName, "a+") do io
                write(io, "[$(i+1)][$nSteps] Current loss is: $l \n")
            end;
            
            # jldsave(saveFile; optParam=Array(p), PDE_losses1, BC_losses1);
        end
        return false
    end

    # set up Optimization optimization problem
    local f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())
    local prob = Optimization.OptimizationProblem(f_, optParams[i])

    println("Calling Optimization() after $(i) iterations of OT sampling")
    local res = Optimization.solve(prob, opt1, callback=cb_, maxiters=maxOpt1Iters);
    local prob = remake(prob, u0=res.minimizer)
    local res = Optimization.solve(prob, opt2, callback=cb_, maxiters=maxOpt2Iters);
    println("Optimization done.")

    optParams[i+1] = res.minimizer

    pde_train_sets_cpu = Vector{Vector{Matrix{Float32}}}(undef, i + 1) 
    for k in 1:i+1    
        pde_train_sets_cpu[k] = Array.(pde_train_sets[k]);
    end

    if runExp
        jldsave(saveFile; optParams = Array.(optParams[1:i+1]), PDE_losses, BC_losses, pde_train_sets = pde_train_sets_cpu, newPtsAll = Array.(newPtsAll[1:i]));
    end
end