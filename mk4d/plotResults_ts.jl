# Plot results for the transient state FPKE for the mk4D problem

cd(@__DIR__);
using JLD2, NeuralPDE, Flux, Trapz, PyPlot, ModelingToolkit, Quadrature
pygui(true);

expNum = 2;
nn = 48; 
activFunc = tanh;
Q_fpke = 0.4f0;
tEnd = 1.0f0;
fileLoc = "data_ts_quasi/ll_quasi_mk4d_exp$(expNum).jld2";
@info "Loading ts results for 4d system from exp $(expNum)";

file = jldopen(fileLoc, "r");
optParam = read(file, "optParam");
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
close(file);
println("Are any of the parameters NaN? $(any(isnan.(optParam)))")

## plot losses
nIters = length(PDE_losses);
figure(1); clf();
semilogy(1:nIters, PDE_losses, label =  "PDE");
semilogy(1:nIters, BC_losses, label = "BC");
xlabel("Iterations"); ylabel("ϵ");
title("Loss Function exp$(expNum)"); legend();
tight_layout();

## set up the NeuralPDE framework using low-level API
@parameters t, x1, x2, x3, x4
@variables  η(..)

xSym = [x1;x2; x3; x4]

# Linear 4D Dynamics
k1 = 1; k2 = 1; k3 = 1;
c1 = 0.4; c2 = 0.4;
linA = Float32.([0 1 0 0;
        -(k1+k2) -c1 k2 0;
        0 0 0 1;
        k2 0 -(k2+k3) -c2]);
f(x) = linA*x;

function g(x::Vector)
    return [0.0f0 0.0f0;1.0f0 0.0f0; 0.0f0 0.0f0; 0.0f0 1.0f0];
end

# PDE
println("Defining PDE");
Q_fpke = 0.4f0; # Q = σ^2
ρ(x) = exp(η(t, x...));

# Equation written directly in η
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G2 = diffC*η(t, xSym...);

dtT = Differential(t)(η(t, xSym...));
T1_2 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(η(t, xSym...)))) for i in 1:length(xSym)]); # drift term
# T2_2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G2[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
# T2_2 += sum([(Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j]) - (Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])*η(xSym...) + diffC[i,j]*(Differential(xSym[i])(η(xSym...)))*(Differential(xSym[j])(η(xSym...))) for i in 1:length(xSym), j=1:length(xSym)]); # complete diffusion term
T2_2 = 0.2f0*abs2(Differential(x2)(η(t, x1, x2, x3, x4))) + 0.2f0*abs2(Differential(x4)(η(t, x1, x2, x3, x4))) + 0.2f0Differential(x2)(Differential(x2)(η(t, x1, x2, x3, x4))) + 0.2f0Differential(x4)(Differential(x4)(η(t, x1, x2, x3, x4))); # diffusion term modified to avoid NaNs when using ADAM

Eqn = expand_derivatives(dtT + T1_2 - T2_2); 
pdeOrig2 = simplify(Eqn, expand = true) ~ 0.0f0;
pde = pdeOrig2;
println("PDE in η defined symbolically.")

## Domains
x1_min = -4.0f0; x1_max = 4.0f0;
x2_min = -4.0f0; x2_max = 4.0f0;
x3_min = -4.0f0; x3_max = 4.0f0;
x4_min = -4.0f0; x4_max = 4.0f0;

maxval = x1_max;
domains = [x1 ∈ IntervalDomain(-maxval,maxval),
           x2 ∈ IntervalDomain(-maxval,maxval),
           x3 ∈ IntervalDomain(-maxval,maxval),
           x4 ∈ IntervalDomain(-maxval,maxval),
           t ∈ IntervalDomain(0.0f0, tEnd)];

bcs = [ρ([-maxval,x2,x3,x4]) ~ 0.f0, ρ([maxval,x2,x3,x4]) ~ 0,
       ρ([x1,-maxval,x3,x4]) ~ 0.f0, ρ([x1,maxval,x3,x4]) ~ 0,
       ρ([x1,x2,-maxval,x4]) ~ 0.f0, ρ([x1,x2,maxval,x4]) ~ 0,
       ρ([x1,x2,x3,-maxval]) ~ 0.f0, ρ([x1,x2,x3,maxval]) ~ 0,];
## Neural network
dim = 5 # number of dimensions
chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));

flat_initθ = optParam;
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

dx = 1.0*ones(dim); # discretization size, to be used for plotting 
strategy = NeuralPDE.GridTraining(dx);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = [t; xSym];
depvars = [η(t, xSym...)];

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

_pde_loss_function = NeuralPDE.build_loss_function(pde, indvars, depvars, phi, derivative, integral, chain, optParam, strategy);

bc_indvars = NeuralPDE.get_variables(bcs, indvars, depvars);
_bc_loss_functions = [NeuralPDE.build_loss_function(bc, indvars, depvars, phi, derivative, integral, chain, optParam, strategy, bc_indvars = bc_indvar) for (bc, bc_indvar) in zip(bcs, bc_indvars)];

@time train_domain_set, train_bound_set = NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);

bc_loss_functions = [NeuralPDE.get_loss_function(loss, set, eltypeθ, parameterless_type_θ, strategy) for (loss, set) in zip(_bc_loss_functions, train_bound_set)];

##
# tx0 = rand(5);
# @show _bc_loss_functions(zeros(4), optParam);
# @show map(l -> l(tx0, optParam), _bc_loss_functions);
bc_loss_function_sum = θ -> sum(map(l -> l(θ), bc_loss_functions))
@show bc_loss_function_sum(optParam)

##
function ρ_pdeErr_fns(optParam)
    function ηNetS(x)
        return first(phi(x, optParam))
    end
    ρNetS(x) = exp(ηNetS(x)) # solution after first iteration
    pdeErrFn(x) = first(_pde_loss_function(x, optParam))

    # bcErrFn(x) =
    
    return ρNetS, pdeErrFn
end
ρFn, pdeErrFn = ρ_pdeErr_fns(optParam);

x1Fine = collect(x1_min:dx[2]:x1_max);
x2Fine = collect(x2_min:dx[3]:x2_max); 
x3Fine = collect(x3_min:dx[4]:x3_max);
x4Fine = collect(x4_min:dx[5]:x4_max); 

function ρ12(t,x1,x2)
    # p = [xV, xα]; y = [xθ, xq]
    ρFnQuad34(y,p) = ρFn([t;x1;x2;y])
    prob = QuadratureProblem(ρFnQuad34, [x3_min; x4_min], [x3_max; x4_max], p = 0)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end

function tr_ρ12(t,x1,x2)
    # using trapz, not quadrature
    ρFn34 = trapz((x3Fine,x4Fine), [ρFn([t;x1;x2;x3d;x4d]) for x3d = x3Fine, x4d = x4Fine]);
    return ρFn34
end

ntFine = 10;
ttFine = range(0.0, tEnd, length = ntFine);

# @info "Computing marginal PDF ρ12";
# RHO12  = [ρ12(tVal,x1d,x2d) for tVal in ttFine, x1d in x1Fine, x2d in x2Fine];
RHO12  = [tr_ρ12(tVal,x1d,x2d) for tVal in ttFine, x1d in x1Fine, x2d in x2Fine];

# Compute only if discretization/grid size is large
# RHOfull  = [ρFn([tVal,x1d,x2d,x3d,x4d]) for tVal in ttFine, x1d in x1Fine, x2d in x2Fine, x3d in x3Fine, x4d in x4Fine];
# @show maximum(RHOfull);
# @show minimum(RHOfull);

##
function plotDistErr(figNum,x1Fine,x2Fine, RHO12, label1, label2)
    XX = zeros(length(x1Fine), length(x2Fine));
    YY = similar(XX);

    for i = 1:length(x1Fine), j = 1:length(x2Fine)
        XX[i, j] = x1Fine[i]
        YY[i, j] = x2Fine[j]
    end

    for (tInd,tVal) in enumerate(ttFine)
    # tInd = 10; tVal = ttFine[tInd]
        figure(figNum); clf();
        pcolor(XX, YY, RHO12[tInd,:,:], shading = "auto", cmap = "inferno"); colorbar();
        # surf(XX, YY, RHO12[tInd,:,:], cmap = "inferno");
        # pcolor(XX, YY, [ρ12(tVal,x1d,x2d) for x1d in x1Fine, x2d in x2Fine], shading = "auto", cmap = "inferno"); colorbar();
        xlabel(label1); ylabel(label2);
        axis("auto")
        title("Marginal PDF ρ($(label1), $(label2))");

        suptitle("t = $(tVal)")
        tight_layout()
        sleep(0.1);
    end
end
plotDistErr(100+expNum, x1Fine,x2Fine, RHO12,  "x1", "x2");

## compare with MOC
# mkpath("figs_moc/exp$(expNum)") # to save figs
using DifferentialEquations, ForwardDiff, Printf, LinearAlgebra
df(x) = ForwardDiff.jacobian(f,x);
uDyn(rho, x) = -tr(df(x)); 
# uDyn(rho,x) = -rho*tr(df(x));
tInt = ttFine[2] - ttFine[1];
function nlSimContMOC(x0)
    odeFn(xu,p,t) = [f(xu[1:4]); uDyn(xu[end], xu[1:4])]
    prob = ODEProblem(odeFn, x0, (0.0, tEnd));
    sol = solve(prob, Tsit5(), saveat = tInt, reltol = 1e-3, abstol = 1e-3);
    return sol.u
end

# RHO0_NN = RHOPred[:,1];
# XU_t = [nlSimContMOC([x, 0.0f0]) for x in xxFine];

# for (tInd, tVal) in enumerate(ttFine)
#     X1grid = [XU_t[i][tInd][1] for i in 1:nEvalFine];
#     RHOgrid_MOC = [RHO0_NN[i]*exp(XU_t[i][tInd][2]) for i in 1:nEvalFine];
#     # RHOgrid_MOC = [(XU_t[i,j][tInd][3]) for i in 1:nEvalFine, j in 1:nEvalFine]; # no change in variabless
#     RHOgrid_NN = [ρFn([XU_t[i][tInd][1], tVal]) for i in 1:nEvalFine];

#     figure(45, (8,4)); clf();
#     subplot(1,2,1);
#     plot(X1grid, RHOgrid_MOC, label = "MOC"); 
#     xlabel("x1"); ylabel("x2");
#     xlim(-maxval, maxval);
#     plot(X1grid, RHOgrid_NN, label = "NN"); 
#     legend();

#     subplot(1,2,2);
#     plot(X1grid, abs.(RHOgrid_MOC - RHOgrid_NN)); 
#     ϵ_mse = sum(abs2, (RHOgrid_MOC - RHOgrid_NN))/(nEvalFine^2);
#     ϵ_mse_str = string(@sprintf "%.2e" ϵ_mse);
#     xlabel("x1"); ylabel("x2");
#     xlim(-maxval, maxval);
#     title("Pointwise ϵ");
#     t_str = string(@sprintf "%.2f" tVal);
#     suptitle("t = $(t_str)")
#     tight_layout();

#     savefig("figs_moc/exp$(expNum)/t$(tInd).png");
#     sleep(0.5);
# end