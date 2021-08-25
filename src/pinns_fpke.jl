module fpke_pinns
## trial module for solving 2d fpke steady state given dynamics f, Q, 
using NeuralPDE,
    Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra

import Random: seed!
seed!(1)

@parameters x1, x2
x = [x1; x2]
@variables η(..)
ρ(x) = exp(η(x[1], x[2]))

function gen_pde(f, Q)
    function g(x::Vector)
        return [0.0; 1.0]
    end
    F = f(x) * ρ(x)
    G = 0.5 * (g(x) * Q * g(x)') * ρ(x)

    T1 = sum([Differential(x[i])(F[i]) for i = 1:length(x)])
    T2 = sum([
        (Differential(x[i]) * Differential(x[j]))(G[i, j]) for i = 1:length(x),
        j = 1:length(x)
    ])

    Eqn = expand_derivatives(-T1 + T2)
    pde = simplify(Eqn / ρ(x), expand = true) ~ 0.0f0
    return pde
end

function gen_domains(maxVal)
    # assuming square grid
    domains = [x1 ∈ IntervalDomain(-maxVal, maxVal), x2 ∈ IntervalDomain(-maxVal, maxVal)]
    return domains
end

function gen_bcs(maxVal)
    # assuming square grid
    bcs = [
        ρ([-maxVal, x2]) ~ 0.0f0,
        ρ([maxVal, x2]) ~ 0,
        ρ([x1, -maxVal]) ~ 0.0f0,
        ρ([x1, maxVal]) ~ 0,
    ]
    return bcs
end

function gen_chain(nn)
    activFunc = tanh
    chain = Chain(Dense(2, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1))
    return chain
end

function solve(f, Q, maxVal, nn, dx, maxOptIters, saveFile)
    pde = gen_pde(f, Q)
    bcs = gen_bcs(maxVal)
    domains = gen_domains(maxVal)
    strategy = NeuralPDE.GridTraining(dx)
    chain = gen_chain(nn)
    opt = Optim.BFGS() # Optimizer used for training

    initθ = DiffEqFlux.initial_params(chain)
    eltypeθ = eltype(initθ)
    parameterless_type_θ = DiffEqBase.parameterless_type(initθ)

    phi = NeuralPDE.get_phi(chain, parameterless_type_θ)
    derivative = NeuralPDE.get_numeric_derivative()

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
    )

    bc_indvars = NeuralPDE.get_variables(bcs, indvars, depvars)
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
        NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars)

    pde_loss_function = NeuralPDE.get_loss_function(
        _pde_loss_function,
        train_domain_set[1],
        eltypeθ,
        parameterless_type_θ,
        strategy,
    )

    bc_loss_functions = [
        NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy)
        for (loss, set) in zip(_bc_loss_functions, train_bound_set)
    ]

    bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))

    function loss_function_(θ, p)
        return pde_loss_function(θ) + bc_loss_function_sum(θ)
    end

    f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
    prob = GalacticOptim.OptimizationProblem(f_, initθ)

    PDE_losses = Float32[]
    BC_losses = Float32[]
    cb_ = function (p, l)
        println("Current loss is: $l")
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

    # println("Calling GalacticOptim()")
    # res = GalacticOptim.solve(prob, opt, cb = cb_, maxiters = maxOptIters)
    # println("Optimization done.")

    ## Save data
    # jldsave(saveFile; optParam = res.minimizer, PDE_losses, BC_losses)
end

end
##


nn = 48; # number of neurons in the hidden layer
dx = 0.05; # discretization size used for training
saveFile = "temp.jld2";
maxOptIters = 10; # maximum number of training iterations

# Van der Pol Dynamics
f(x) = [x[2]; -x[1] + (1 - x[1]^2) * x[2]];
Q = 0.1; # Q = σ^2
maxVal = 4.0;

fpke_pinns.solve(f, Q, maxVal, nn, dx, maxOptIters, saveFile)

## 
using BenchmarkTools
@btime fpke_pinns.solve(f, Q, maxVal, nn, dx, maxOptIters, saveFile);
