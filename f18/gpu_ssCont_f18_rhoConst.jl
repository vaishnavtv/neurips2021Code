## Obtain a controller for f18 using PINNs
# terminal state pdf is fixed - using a thin gaussian about origin
cd(@__DIR__);
include("f18DynNorm.jl") # normalized state variable info
include("f18Dyn.jl")
mkpath("out_rhoConst_gpu")
mkpath("data_rhoConst_gpu")

using NeuralPDE, Flux, ModelingToolkit, Optimization, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra, Distributions

import Random: seed!;
seed!(1);

## parameters for neural network
nn = 100; # number of neurons in the hidden layer
activFunc = tanh; # activation function
opt1 = ADAM(1e-3); # primary optimizer used for training
maxOpt1Iters = 10000; # maximum number of training iterations for opt1
opt2 = Optim.LBFGS(); # second optimizer used for fine-tuning
maxOpt2Iters = 10000; # maximum number of training iterations for opt2

# parameters for rhoSS_desired
# μ_ss = [0f0,0f0,0f0,0f0] #.+ Array(f18_xTrim[indX]);
# Σ_ss = 0.01f0*Array(f18_xTrim[indX]).*1.0f0I(4);
μ_ss = An*([0f0,0f0,0f0,0f0] .+ Array(f18_xTrim[indX])) + bn;
Σ_ss = 0.01f0.*1.0f0I(4);
# indU = [3]; # only using δ_stab for control

Q_fpke = 0.0f0; # Q = σ^2
dx = 0.01f0;

# file location to save data
expNum = 9;
useGPU = false;
runExp = true;
saveFile = "data_rhoConst_gpu/exp$(expNum).jld2";
runExp_fileName = "out_rhoConst_gpu/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Generating a controller for f18 with desired ss distribution. 2 HL with $(nn) neurons in the hl and $(activFunc) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS. using GPU? $(useGPU). Q_fpke = $(Q_fpke). μ_ss = $(μ_ss). Σ_ss = $(Σ_ss). Not dividing equation by ρ. Finding utrim, using xN as input. Changed several things. useGPU = $(useGPU). Changed Σ_ss. dx = $(dx).
        Experiment number: $(expNum)\n")
    end
end

## set up the NeuralPDE framework using low-level API
@parameters x1, x2, x3, x4
@variables Kc1(..), Kc2(..)

xSym = [x1; x2; x3; x4]

##
maskIndx = zeros(Float32,(length(f18_xTrim),length(indX)));
maskIndu = zeros(Float32,(length(f18_uTrim),length(indU)));
for i in 1:length(indX)
    maskIndx[indX[i],i] = 1f0;
    if i<=length(indU)
        maskIndu[indU[i],i] = 1f0;
    end
end

# F18 Dynamics
maskTrim = ones(Float32,length(f18_xTrim)); maskTrim[indX] .= 0f0;
function f(xn)

    # # ud = Kc1(xd[1],xd[2],xd[3],xd[4]); 
    # ud = [Kc1(xd[1],xd[2],xd[3],xd[4]); Kc2(xd[1],xd[2],xd[3],xd[4])];

    # # tx = ((maskIndx)*xd); tu = ((maskIndu)*ud);
    # # xFull = Vector{Real}(undef, 9);
    # # uFull = Vector{Real}(undef, 4);
    # # for i in 1:9
    # #     xFull[i] = f18_xTrim[i] + tx[i];
    # # end 
    # # for i in 1:4
    # #     uFull[i] = f18_uTrim[i] + tu[i];
    # # end 
    # # perturbation about trim point
    # xFull = f18_xTrim + maskIndx*xd; 
    # maskTrim = ones(Float32,length(f18_xTrim)); maskTrim[indX] .= 0f0;
    # # xFull = maskTrim.*f18_xTrim + maskIndx*xd; 
    # uFull = [1f0;1f0;0f0;1f0].*f18_uTrim + maskIndu*ud;

    # xdotFull = f18Dyn(xFull, uFull)
    # # xdotFull = xFull;

    # return (xdotFull[indX]) # return the 4 state dynamics

    # normalized input to f18 dynamics (full dynamics)
    xi = An\(xn-bn); # x of 'i'nterest
    ui = [Kc1(xn...), Kc2(xn...)];

    xFull = maskTrim.*f18_xTrim + maskIndx*xi;
    uFull = [1f0;1f0;0f0;0f0].*f18_uTrim + maskIndu*ui; 

    xdotFull = f18Dyn(xFull, uFull)
    return An*(xdotFull[indX]) # return the 4 state dynamics in normalized form

end

##
g(x::Vector) = [1.0f0; 1.0f0;1.0f0; 1.0f0] # diffusion vector needs to be modified

# PDE
ρSS_sym = pdf(MvNormal(μ_ss, Σ_ss),xSym);
F = f(xSym) * ρSS_sym;
G = 0.5f0 * (g(xSym) * Q_fpke * g(xSym)') * ρSS_sym;

# driftTerm = sum([Differential(xSym[i])(F[i]) for i = 1:length(xSym)]);
T1 = Differential(xSym[1])(F[1]) #length(xSym)]);
T2 = Differential(xSym[2])(F[2]) #length(xSym)]);
T3 = Differential(xSym[3])(F[3]) #length(xSym)]);
T4 = Differential(xSym[4])(F[4]) #length(xSym)]);
# T2 = sum([(Differential(xSym[i]) * Differential(xSym[j]))(G[i, j]) for i = 1:length(xSym),j = 1:length(xSym)]);

# Eqn = expand_derivatives(-T1 + T2); # + dx*u(x1,x2)-1 ~ 0;
# pde = simplify(Eqn) ~ 0.0f0;
pde = [expand_derivatives(T1) ~ 0.f0, expand_derivatives(T2) ~ 0.0f0, expand_derivatives(T4) ~ 0.0f0]; # T3 not dependent on Kc, will sum these terms later
# pde = driftTerm ~ 0.0f0
println("PDE defined.")

## Domain
# All xi between 0 and 1
x1_min = vN(f18_xTrim[indX[1]] - 100f0) ; x1_max = vN(f18_xTrim[indX[1]] + 100f0) #+ f18_xTrim[indX[1]];
x2_min = alpN(f18_xTrim[indX[2]] - deg2rad(10f0)) ; x2_max = alpN(f18_xTrim[indX[2]] + deg2rad(10f0)) #+ f18_xTrim[indX[2]];
x3_min = x2_min ; x3_max = x2_max #+ f18_xTrim[indX[3]];
x4_min = qN(f18_xTrim[indX[4]] + deg2rad(-5f0)) ; x4_max = qN(f18_xTrim[indX[4]] + deg2rad(5f0)) #+ f18_xTrim[indX[4]];
domains = [x1 ∈ IntervalDomain(x1_min, x1_max), x2 ∈ IntervalDomain(x2_min, x2_max), x3 ∈ IntervalDomain(x3_min, x3_max), x4 ∈ IntervalDomain(x4_min, x4_max),];

# dx = [10f0; deg2rad(1f0); deg2rad(1f0); deg2rad(1f0);]; # discretization size used for training

# Boundary conditions
bcs = [Kc1(x1_min,x2,x3,x4) ~ 0.f0, Kc2(100f0,x2,x3,x4) ~ 0.f0]; # place holder, not really used


## Neural network set up
dim = 4 # number of dimensions
chain1 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
# chain2 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain = chain1;#[chain1, chain2];

initθ = DiffEqFlux.initial_params(chain);
if useGPU
    using CUDA
    CUDA.allowscalar(false)
    initθ = initθ |> gpu;
    th10= initθ#[1];
    # th20= initθ[2];
end 
flat_initθ = initθ;#reduce(vcat, initθ); 
th0 = flat_initθ;
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);


strategy = NeuralPDE.GridTraining(dx);

indvars = xSym
depvars = [Kc1(xSym...), Kc2(xSym...)]

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();
integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);
## Loss function
println("Defining loss function for each term.")
_pde_loss_functions = [NeuralPDE.build_loss_function(pde_i, indvars, depvars, phi, derivative, integral, chain, initθ, strategy) for pde_i in pde];
# _pde_loss_function = NeuralPDE.build_loss_function(pde, indvars, depvars, phi, derivative, integral, chain, initθ, strategy);
# _pde_loss_function(tx, th0) # ptxas code issue
tx = (μ_ss);
@show [fn(tx, th0) for fn in _pde_loss_functions]
_pde_loss_function2(cord, θ) = sum([fn(cord, θ) for fn in _pde_loss_functions]);

train_domain_set, train_bound_set =
    NeuralPDE.generate_training_sets(domains, dx, pde, bcs, eltypeθ, indvars, depvars);
if useGPU
    train_domain_set = train_domain_set |> gpu;
    train_bound_set = train_bound_set |> gpu;
end

using Statistics
pde_loss_function = (θ) -> mean(abs2,_pde_loss_function2((train_domain_set[1]), θ));
@show pde_loss_function(flat_initθ)

loss_function_(θ, p) =  pde_loss_function(θ)

## set up Optimization optimization problem
f_ = OptimizationFunction(loss_function_, Optimization.AutoZygote())
prob = Optimization.OptimizationProblem(f_, flat_initθ)

nSteps = 0;
PDE_losses = Float32[];
cb_ = function (p, l)
    if any(isnan.(p))
        println("SOME PARAMETERS ARE NaN.")
    end

    global nSteps = nSteps + 1
    println("[$nSteps] Current loss is: $l")

    push!(PDE_losses, l)

    if runExp # if running job file
        open(runExp_fileName, "a+") do io
            write(io, "[$nSteps] Current loss is: $l \n")
        end;
        
        jldsave(saveFile; optParam=Array(p), PDE_losses);
    end
    return false
end

println("Calling Optimization()");
res = Optimization.solve(prob, opt1, callback=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer);
res = Optimization.solve(prob, opt2, callback=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

## Save data
if runExp
    jldsave(saveFile;optParam = Array(res.minimizer),PDE_losses);
end

##
# bc_indvars = NeuralPDE.get_argument(bcs, indvars, depvars);
# depvars,indvars,dict_indvars,dict_depvars, dict_depvar_input = NeuralPDE.get_vars(indvars, depvars);
# symLoss = [NeuralPDE.build_symbolic_loss_function(pde[i],indvars,depvars,
# dict_indvars,dict_depvars,dict_depvar_input, phi,derivative,integral,chain,initθ,strategy, param_estim = false, bc_indvars = bc_indvars, eq_params = SciMLBase.NullParameters(), default_p = nothing) for i in 1:4];
# #
# symLoss1 = ((cord, var"##θ#292", phi, derivative, integral, u, p)->begin
# begin
#     (var"##θ#2921", var"##θ#2922") = (var"##θ#292"[1:10701], var"##θ#292"[10702:21402])
#     (phi1, phi2) = (phi[1], phi[2])
#     let (x1, x2, x3, x4) = (cord[[1], :], cord[[2], :], cord[[3], :], cord[[4], :])
#         begin
#             cord2 = vcat(x1, x2, x3, x4)
#             cord1 = vcat(x1, x2, x3, x4)
#         end
#     end
# end
# end)

# # #
# tx = cu(μ_ss);
# uPhi = NeuralPDE.get_u();
# _loss_function = symLoss1;
# _pde_loss_function = (cord, θ) -> begin
#     _loss_function(cord, θ, phi, derivative, integral, uPhi, nothing)
# end

# @show _pde_loss_function((tx), th0)
# @show _pde_loss_function((tx1), th0)
##
# symb_eq1 = NeuralPDE.parse_equation(pde[1],indvars,depvars,dict_indvars,dict_depvars,dict_depvar_input,chain,eltypeθ,strategy,phi,derivative,integral,initθ);
# eq1_lhs = isequal(expand_derivatives(pde[1].lhs), 0) ? pde[1].lhs : expand_derivatives(pde[1].lhs);
# eq1_rhs = isequal(expand_derivatives(pde[1].rhs), 0) ? pde[1].rhs : expand_derivatives(eq.rhs);
# eq1_lexpr = NeuralPDE.transform_expression(toexpr(eq1_lhs),indvars,depvars,dict_indvars,dict_depvars,dict_depvar_input,chain,eltypeθ,strategy,phi,derivative,integral,initθ);
# eq1_lexpr_d = NeuralPDE._dot_(eq1_lexpr)

# # eq1_l2 = :(eq1_lexpr - 0f0);

# lTmp = ((cord, var"##θ#292", phi, derivative, integral, u, p)->begin
# begin
#     (var"##θ#2921", var"##θ#2922") = (var"##θ#292"[1:10701], var"##θ#292"[10702:21402])
#     (phi1, phi2) = (phi[1], phi[2])
#     let (x1, x2, x3, x4) = (cord[[1], :], cord[[2], :], cord[[3], :], cord[[4], :])
#         begin
#             cord2 = vcat(x1, x2, x3, x4)
#             cord1 = vcat(x1, x2, x3, x4)
#         end
#
#     end
# end
# end)

# uPhi = NeuralPDE.get_u();
# _pde_loss_function = (cord, θ) -> begin
#     lTmp(cord, θ, phi, derivative, integral, uPhi, nothing)
# end

# tx = cu(μ_ss);
# tx1 = train_domain_set[1][:,100:105]

# @show _pde_loss_function((tx), th0)
# @show _pde_loss_function((tx1), th0)