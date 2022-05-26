## Obtain a controller for f18 using PINNs
# terminal state pdf is fixed - using a thin gaussian about origin
cd(@__DIR__);
include("f18Dyn.jl")
mkpath("out_rhoConst_gpu")
mkpath("data_rhoConst_gpu")

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, Symbolics, JLD2, DiffEqFlux, LinearAlgebra, Distributions

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
μ_ss = [0f0,0f0,0f0,0f0];
Σ_ss = 0.1f0*Array(f18_xTrim[indX]).*1.0f0I(4);

Q_fpke = 0.0f0; # Q = σ^2

# file location to save data
expNum = 1;
useGPU = true;
runExp = true;
saveFile = "data_rhoConst_gpu/exp$(expNum).jld2";
runExp_fileName = "out_rhoConst_gpu/log$(expNum).txt";
if runExp
    open(runExp_fileName, "a+") do io
        write(io, "Generating a controller for f18 with desired ss distribution. 2 HL with $(nn) neurons in the hl and $(activFunc) activation. $(maxOpt1Iters) iterations with ADAM and then $(maxOpt2Iters) with LBFGS. using GPU? $(useGPU). Q_fpke = $(Q_fpke). μ_ss = $(μ_ss). Σ_ss = $(Σ_ss). Not dividing equation by ρ. Manually wrote loss function.
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

if useGPU
    f18_xTrim = f18_xTrim |> gpu;
    f18_uTrim = f18_uTrim |> gpu;
    maskIndx = maskIndx |> gpu;
    maskIndu = maskIndu |> gpu;
else
    f18_xTrim = f18_xTrim |> cpu;
    f18_uTrim = f18_uTrim |> cpu;
    maskIndx = maskIndx |> cpu;
    maskIndu = maskIndu |> cpu;
end

# F18 Dynamics
function f(xd)

    ud = [Kc1(xd[1],xd[2],xd[3],xd[4]); Kc2(xd[1],xd[2],xd[3],xd[4])];

    tx = ((maskIndx)*xd); tu = ((maskIndu)*ud);
    xFull = Vector{Real}(undef, 9);
    uFull = Vector{Real}(undef, 4);
    for i in 1:9
        xFull[i] = f18_xTrim[i] + tx[i];
    end 
    for i in 1:4
        uFull[i] = f18_uTrim[i] + tu[i];
    end 
    # perturbation about trim point
    # xFull = f18_xTrim + maskIndx*xd; 
    # uFull = f18_uTrim + maskIndu*ud;

    xdotFull = f18Dyn(xFull, uFull)
    # xdotFull = xFull;

    return (xdotFull[indX]) # return the 4 state dynamics

end

##
g(x::Vector) = [1.0f0; 1.0f0;1.0f0; 1.0f0] # diffusion vector needs to be modified

# PDE
ρSS_sym = pdf(MvNormal(μ_ss, Σ_ss),xSym);
F = f(xSym) * ρSS_sym;
G = 0.5f0 * (g(xSym) * Q_fpke * g(xSym)') * ρSS_sym;

# T1 = sum([Differential(xSym[i])(F[i]) for i = 1:length(xSym)]);
T1 = Differential(xSym[1])(F[1]) #length(xSym)]);
T2 = Differential(xSym[2])(F[2]) #length(xSym)]);
T3 = Differential(xSym[3])(F[3]) #length(xSym)]);
T4 = Differential(xSym[4])(F[4]) #length(xSym)]);
# T2 = sum([(Differential(xSym[i]) * Differential(xSym[j]))(G[i, j]) for i = 1:length(xSym),j = 1:length(xSym)]);

# Eqn = expand_derivatives(-T1 + T2); # + dx*u(x1,x2)-1 ~ 0;
# pde = simplify(Eqn) ~ 0.0f0;
pde = [T1 ~ 0.f0, T2 ~ 0.0f0, T3 ~ 0.0f0, T4 ~ 0.0f0];

println("PDE defined.")

## Domain
x1_min = -100f0; x1_max = 100f0;
x2_min = deg2rad(-10f0); x2_max = deg2rad(10f0);
x3_min = x2_min; x3_max = x2_max;
x4_min = deg2rad(-5f0); x4_max = deg2rad(5f0);
domains = [x1 ∈ IntervalDomain(x1_min, x1_max), x2 ∈ IntervalDomain(x2_min, x2_max), x3 ∈ IntervalDomain(x3_min, x3_max), x4 ∈ IntervalDomain(x4_min, x4_max),];

dx = [10f0; deg2rad(1f0); deg2rad(1f0); deg2rad(1f0);]; # discretization size used for training

# Boundary conditions
bcs = [Kc1(-100f0,x2,x3,x4) ~ 0.f0, Kc2(100f0,x2,x3,x4) ~ 0.f0]; # place holder, not really used


## Neural network set up
dim = 4 # number of dimensions
chain1 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain2 = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain = [chain1, chain2];

initθ = DiffEqFlux.initial_params.(chain);
if useGPU
    using CUDA
    CUDA.allowscalar(false)
    initθ = initθ |> gpu;
    th10= initθ[1];
    th20= initθ[2];
end 
flat_initθ = reduce(vcat, initθ); 
th0 = flat_initθ;
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);


strategy = NeuralPDE.GridTraining(dx);

indvars = xSym
depvars = [Kc1(xSym...), Kc2(xSym...)]

phi = NeuralPDE.get_phi.(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();
integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);
## Loss function
println("Defining loss function.")
sleep(10000);
# _pde_loss_function = NeuralPDE.build_loss_function(pde, indvars, depvars, phi, derivative, integral, chain, initθ, strategy);
# tx = cu(μ_ss);
# _pde_loss_function(tx, th0)

symLoss1 = ((cord, var"##θ#292", phi, derivative, integral, u, p)->begin
begin
    (var"##θ#2921", var"##θ#2922") = (var"##θ#292"[1:10701], var"##θ#292"[10702:21402])
    (phi1, phi2) = (phi[1], phi[2])
    let (x1, x2, x3, x4) = (cord[[1], :], cord[[2], :], cord[[3], :], cord[[4], :])
        begin
            cord2 = vcat(x1, x2, x3, x4)
            cord1 = vcat(x1, x2, x3, x4)
        end
        (+)((*).((+).((+).((*).((+).(700.0f0, (*).(2, x1)), (+).((+).((+).((+).((+).(1.36931f-5, (*).(4.111493f-5, x2)), (*).(-0.0013183776f0, (^).((+).(0.3540971f0, x2), 2))), (*).(-0.00030109732f0, (^).((+).(0.3540971f0, x2), 4))), (*).(0.00118174f0, (^).((+).(0.3540971f0, x2), 3))), (*).((*).(-0.00020608988f0, (+).(-0.078592755f0, u(cord1, var"##θ#2921", phi1))), (+).((+).((+).(-0.060387194f0, (*).(-0.2739f0, x2)), (*).(4.236f0, (^).((+).(0.3540971f0, x2), 2))), (*).(-3.8578f0, (^).((+).(0.3540971f0, x2), 3)))))), (*).((*).(0.00096665055f0, (cos).((+).(0.3540971f0, x2))), derivative(phi2, u, cord2, Vector{Float32}[[0.0049215667, 0.0, 0.0, 0.0]], 1, var"##θ#2922"))), (*).((*).((*).(-0.00020608988f0, (+).((+).((+).(-0.060387194f0, (*).(-0.2739f0, x2)), (*).(4.236f0, (^).((+).(0.3540971f0, x2), 2))), (*).(-3.8578f0, (^).((+).(0.3540971f0, x2), 3)))), (^).((+).(350.0f0, x1), 2)), derivative(phi1, u, cord1, Vector{Float32}[[0.0049215667, 0.0, 0.0, 0.0]], 1, var"##θ#2921"))), (exp).((+).((+).((+).((+).(0.79735756f0, (*).(-1//2, (abs2).((*).(0.16903085f0, x1)))), (*).(-1//2, (abs2).((*).(5.3142114f0, x2)))), (*).(-1//2, (abs2).((*).(5.536393f0, x3)))), (*).(-1//2, (abs2).((*).(17.620409f0, x4)))))), (*).((*).((*).(-0.028571427f0, x1), (+).((+).((+).((*).((+).(14.016433f0, (*).(0.00096665055f0, u(cord2, var"##θ#2922", phi2))), (cos).((+).(0.3540971f0, x2))), (*).((*).(-0.00020608988f0, (+).((+).((+).((+).((+).(-0.06644237f0, (*).(-0.1995f0, x2)), (*).((+).(-0.078592755f0, u(cord1, var"##θ#2921", phi1)), (+).((+).((+).(-0.060387194f0, (*).(-0.2739f0, x2)), (*).(4.236f0, (^).((+).(0.3540971f0, x2), 2))), (*).(-3.8578f0, (^).((+).(0.3540971f0, x2), 3))))), (*).(6.3971f0, (^).((+).(0.3540971f0, x2), 2))), (*).(1.461f0, (^).((+).(0.3540971f0, x2), 4))), (*).(-5.7341f0, (^).((+).(0.3540971f0, x2), 3)))), (^).((+).(350.0f0, x1), 2))), (*).((*).(26.376698f0, (sin).((+).(0.3540971f0, x2))), (cos).((+).(0.32624668f0, x3)))), (*).((*).(-32.2f0, (cos).((+).(0.3540971f0, x2))), (sin).((+).(0.32624668f0, x3))))), (exp).((+).((+).((+).((+).(0.79735756f0, (*).(-1//2, (abs2).((*).(0.16903085f0, x1)))), (*).(-1//2, (abs2).((*).(5.3142114f0, x2)))), (*).(-1//2, (abs2).((*).(5.536393f0, x3)))), (*).(-1//2, (abs2).((*).(17.620409f0, x4))))))) #.- 0.0f0
        (+)((*).((+).((+).((*).((+).(-0.07213146f0, (*).(-0.00020608988f0, x1)), (+).((+).((+).((+).(1.8353298f0, (*).(-10.8492f0, x2)), (*).(3.4935f0, (^).((+).(0.3540971f0, x2), 2))), (*).((+).(-0.078592755f0, u(cord1, var"##θ#2921", phi1)), (+).((+).(-1.5048537f0, (*).(-5.395f0, x2)), (*).(6.5556f0, (^).((+).(0.3540971f0, x2), 2))))), (*).((+).((+).((+).(0.7160864f0, (*).(0.4055f0, x2)), (*).(-2.6975f0, (^).((+).(0.3540971f0, x2), 2))), (*).(2.1852f0, (^).((+).(0.3540971f0, x2), 3))), derivative(phi1, u, cord1, Vector{Float32}[[0.0, 0.0049215667, 0.0, 0.0]], 1, var"##θ#2921")))), (/).((+).((*).((*).(-26.376698f0, (sin).((+).(0.3540971f0, x2))), (cos).((+).(0.32624668f0, x3))), (*).((*).(32.2f0, (cos).((+).(0.3540971f0, x2))), (sin).((+).(0.32624668f0, x3)))), (+).(350.0f0, x1))), (/).((+).((*).((+).(-14.016433f0, (*).(-0.00096665055f0, u(cord2, var"##θ#2922", phi2))), (cos).((+).(0.3540971f0, x2))), (*).((*).(-0.00096665055f0, (sin).((+).(0.3540971f0, x2))), derivative(phi2, u, cord2, Vector{Float32}[[0.0, 0.0049215667, 0.0, 0.0]], 1, var"##θ#2922"))), (+).(350.0f0, x1))), (exp).((+).((+).((+).((+).(0.79735756f0, (*).(-1//2, (abs2).((*).(0.16903085f0, x1)))), (*).(-1//2, (abs2).((*).(5.3142114f0, x2)))), (*).(-1//2, (abs2).((*).(5.536393f0, x3)))), (*).(-1//2, (abs2).((*).(17.620409f0, x4)))))), (*).((*).((*).(-28.240843f0, x2), (+).((+).((+).((+).(0.032208312f0, x4), (/).((+).((*).((*).(32.2f0, (sin).((+).(0.3540971f0, x2))), (sin).((+).(0.32624668f0, x3))), (*).((*).(26.376698f0, (cos).((+).(0.32624668f0, x3))), (cos).((+).(0.3540971f0, x2)))), (+).(350.0f0, x1))), (/).((*).((*).(-1, (+).(14.016433f0, (*).(0.00096665055f0, u(cord2, var"##θ#2922", phi2)))), (sin).((+).(0.3540971f0, x2))), (+).(350.0f0, x1))), (*).((*).(-0.00020608988f0, (+).(350.0f0, x1)), (+).((+).((+).((+).(1.9898093f0, (*).(5.677f0, x2)), (*).((+).(-0.078592755f0, u(cord1, var"##θ#2921", phi1)), (+).((+).((+).(0.7160864f0, (*).(0.4055f0, x2)), (*).(-2.6975f0, (^).((+).(0.3540971f0, x2), 2))), (*).(2.1852f0, (^).((+).(0.3540971f0, x2), 3))))), (*).(-5.4246f0, (^).((+).(0.3540971f0, x2), 2))), (*).(1.1645f0, (^).((+).(0.3540971f0, x2), 3)))))), (exp).((+).((+).((+).((+).(0.79735756f0, (*).(-1//2, (abs2).((*).(0.16903085f0, x1)))), (*).(-1//2, (abs2).((*).(5.3142114f0, x2)))), (*).(-1//2, (abs2).((*).(5.536393f0, x3)))), (*).(-1//2, (abs2).((*).(17.620409f0, x4))))))) #.- 0.0f0
        (+)((*).((*).((*).(-30.65165f0, x3), (+).(1.8626451f-9, (*).(0.81915206f0, x4))), (exp).((+).((+).((+).((+).(0.79735756f0, (*).(-1//2, (abs2).((*).(0.16903085f0, x1)))), (*).(-1//2, (abs2).((*).(5.3142114f0, x2)))), (*).(-1//2, (abs2).((*).(5.536393f0, x3)))), (*).(-1//2, (abs2).((*).(17.620409f0, x4))))))) #.- 0.0f0
        (+)((*).((*).((*).(1.6233826f-5, (^).((+).(350.0f0, x1), 2)), (+).((*).((+).((+).(-1.0200045f0, (*).(-0.3245f0, x2)), (*).(0.9338f0, (^).((+).(0.3540971f0, x2), 2))), derivative(phi1, u, cord1, Vector{Float32}[[0.0, 0.0, 0.0, 0.0049215667]], 1, var"##θ#2921")), (/).((+).((+).((+).(-1.3036569f0, (*).(63.3145f0, x2)), (*).(-394.92923f0, (^).((+).(0.3540971f0, x2), 2))), (*).(372.78146f0, (^).((+).(0.3540971f0, x2), 3))), (+).(350.0f0, x1)))), (exp).((+).((+).((+).((+).(0.79735756f0, (*).(-1//2, (abs2).((*).(0.16903085f0, x1)))), (*).(-1//2, (abs2).((*).(5.3142114f0, x2)))), (*).(-1//2, (abs2).((*).(5.536393f0, x3)))), (*).(-1//2, (abs2).((*).(17.620409f0, x4)))))), (*).((*).((*).(-310.47882f0, x4), (+).(-0.0008832563f0, (*).((*).(1.6233826f-5, (+).((+).((+).((+).(0.09434361f0, (*).(0.511f0, x2)), (*).(-1.2897f0, (^).((+).(0.3540971f0, x2), 2))), (*).((+).(-0.078592755f0, u(cord1, var"##θ#2921", phi1)), (+).((+).(-1.0200045f0, (*).(-0.3245f0, x2)), (*).(0.9338f0, (^).((+).(0.3540971f0, x2), 2))))), (/).((*).((*).(1//2, (+).(0.032208312f0, x4)), (+).((+).((+).(-2.6073139f0, (*).(126.629f0, x2)), (*).(-789.85846f0, (^).((+).(0.3540971f0, x2), 2))), (*).(745.5629f0, (^).((+).(0.3540971f0, x2), 3)))), (+).(350.0f0, x1)))), (^).((+).(350.0f0, x1), 2)))), (exp).((+).((+).((+).((+).(0.79735756f0, (*).(-1//2, (abs2).((*).(0.16903085f0, x1)))), (*).(-1//2, (abs2).((*).(5.3142114f0, x2)))), (*).(-1//2, (abs2).((*).(5.536393f0, x3)))), (*).(-1//2, (abs2).((*).(17.620409f0, x4))))))) .- 0.0f0
    end
end
end)

uPhi = NeuralPDE.get_u();
_loss_function = symLoss1; 
_pde_loss_function = (cord, θ) -> begin
    _loss_function(cord, θ, phi, derivative, integral, uPhi, nothing)
end

##
train_domain_set, train_bound_set =
    NeuralPDE.generate_training_sets(domains, dx, pde, bcs, eltypeθ, indvars, depvars);
if useGPU
    train_domain_set = train_domain_set |> gpu;
    train_bound_set = train_bound_set |> gpu;
end

using Statistics
pde_loss_function = (θ) -> mean(abs2,_pde_loss_function(train_domain_set[1], θ));
@show pde_loss_function(flat_initθ)

loss_function_(θ, p) =  pde_loss_function(θ)

## set up GalacticOptim optimization problem
f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, flat_initθ)

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

println("Calling GalacticOptim()");
res = GalacticOptim.solve(prob, opt1, callback=cb_, maxiters=maxOpt1Iters);
prob = remake(prob, u0=res.minimizer);
res = GalacticOptim.solve(prob, opt2, callback=cb_, maxiters=maxOpt2Iters);
println("Optimization done.");

## Save data
if runExp
    jldsave(saveFile;optParam = Array(res.minimizer),PDE_losses);
end

##
# bc_indvars = NeuralPDE.get_argument(bcs, indvars, depvars);
# depvars,indvars,dict_indvars,dict_depvars, dict_depvar_input = NeuralPDE.get_vars(indvars, depvars);
# symLoss = [NeuralPDE.build_symbolic_loss_function(pde[i],indvars,depvars,
# dict_indvars,dict_depvars,dict_depvar_input,
# phi,derivative,integral,chain,initθ,strategy, param_estim = false, bc_indvars = bc_indvars, eq_params = SciMLBase.NullParameters(), default_p = nothing) for i in 1:4];
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

# #
# tx = cu(μ_ss);
# uPhi = NeuralPDE.get_u();
# _loss_function = symLoss1;
# _pde_loss_function = (cord, θ) -> begin
#     _loss_function(cord, θ, phi, derivative, integral, uPhi, nothing)
# end

# _pde_loss_function((tx), th0)
