# Plot results for the transient state FPKE for the 1D problem

cd(@__DIR__);
using JLD2, NeuralPDE, Flux, Trapz, PyPlot
pygui(true);

expNum = 1;
nn = 48; 
activFunc = tanh;
Q_fpke = 0.25;
tEnd = 0.1f0;
fileLoc = "data_ts_grid/eta_exp$(expNum).jld2";
@info "Loading ts results for 1d system from exp $(expNum)";

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
@parameters x1, t
@variables  η(..)

xSym = x1;

# 1D Dynamics
α = 0.3f0; β = 0.5f0;
f(x) = α*x - β*x^3
g(x) = 1.0f0;

# PDE
ρ(x) = exp(η(x1,t));
F = f(xSym)*ρ(xSym);
#  PDE written directly in η
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G = diffC*η(x1,t);

dtT = Differential(t)(η(x1,t));
T1 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(η(x1,t)))) for i in 1:length(xSym)]); # drift term
T2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
T2 += sum([(Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])  - (Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])*η(x1,t) + diffC[i,j]*abs2(Differential(xSym[i])(η(x1,t))) for i in 1:length(xSym), j=1:length(xSym)]); # complete diffusion term, modified for GPU

Eqn = expand_derivatives(dtT + T1-T2); 
pdeOrig = simplify(Eqn, expand = true) ~ 0.0f0;
pde = pdeOrig;

## Domain
maxval = 2.2f0; tEnd = 0.1f0;
domains = [x1 ∈ IntervalDomain(-maxval,maxval),
            t ∈ IntervalDomain(0.0f0, tEnd)];

## Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1));

flat_initθ = optParam;
eltypeθ = eltype(flat_initθ);
parameterless_type_θ = DiffEqBase.parameterless_type(flat_initθ);

strategy = NeuralPDE.GridTraining(dx);

phi = NeuralPDE.get_phi(chain, parameterless_type_θ);
derivative = NeuralPDE.get_numeric_derivative();

indvars = [x1, t]
depvars = [η(x1, t)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative);

_pde_loss_function = NeuralPDE.build_loss_function(
    pde,
    indvars,
    depvars,
    phi,
    derivative,
    integral,
    chain,
    optParam,
    strategy,
);

train_domain_set, train_bound_set =
    NeuralPDE.generate_training_sets(domains, dx, [pde], bcs, eltypeθ, indvars, depvars);

pde_loss_function = NeuralPDE.get_loss_function(
    _pde_loss_function,
    train_domain_set[1],
    eltypeθ,
    parameterless_type_θ,
    strategy,
);

function ρ_pdeErr_fns(optParam)
    function ηNetS(x)
        return first(phi(x, optParam))
    end
    ρNetS(x) = exp(ηNetS(x)) # solution after first iteration
    pdeErrFn(x) = first(_pde_loss_function(x, optParam))
    
    return ρNetS, pdeErrFn
end
ρFn, pdeErrFn = ρ_pdeErr_fns(optParam);

nEvalFine = 100;
ntFine = 10;
xxFine = range(-maxval, maxval, length = nEvalFine);
ttFine = range(0.0, tEnd, length = ntFine);

RHOPred = [ρFn([x, t]) for x in xxFine, t in ttFine];
# pdeErrFine = [pdeErrFn([x, t])^2 for x in xxFine, t in ttFine]; # squared equation error on the evaluation grid
##
function plotDistErr(figNum)
    # tInd = 3
    for (tInd,tVal) in enumerate(ttFine)
    # tInd = 10; tVal = ttFine[tInd]
        figure(figNum, [8, 4])
        clf()
        subplot(1, 2, 1)
        plot(xxFine, [ρFn([x, tVal]) for x in xxFine]); #colorbar();
        xlabel("x")
        ylabel("ρ");
        axis("auto")
        title("PDF (ρ)")

        subplot(1, 2, 2)
        plot(xxFine, abs.([pdeErrFn([x, tVal]) for x in xxFine]));
        title("Pointwise PDE Error")
        # colorbar()
        axis("auto")
        # title("Pointwise PDE Error")
        # title(L"Equation Error; $ϵ_{pde}$ = %$(mseEqErrStr)")
        xlabel("x1")
        ylabel("x2")
        suptitle("t = $(tVal)")
        tight_layout()
        sleep(0.1);
    end
end
plotDistErr(expNum+100);

## compare with MOC
mkpath("figs_moc/exp$(expNum)") # to save figs
using DifferentialEquations, ForwardDiff, Printf
df(x) = ForwardDiff.derivative(f,x);
uDyn(rho, x) = -tr(df(x)); 
# uDyn(rho,x) = -rho*tr(df(x));
tInt = ttFine[2] - ttFine[1];
function nlSimContMOC(x0)
    odeFn(xu,p,t) = [f(xu[1]); uDyn(xu[end], xu[1])]
    prob = ODEProblem(odeFn, x0, (0.0, tEnd));
    sol = solve(prob, Tsit5(), saveat = tInt, reltol = 1e-3, abstol = 1e-3);
    return sol.u
end

RHO0_NN = RHOPred[:,1];
XU_t = [nlSimContMOC([x, 0.0f0]) for x in xxFine];

for (tInd, tVal) in enumerate(ttFine)
    X1grid = [XU_t[i][tInd][1] for i in 1:nEvalFine];
    RHOgrid_MOC = [RHO0_NN[i]*exp(XU_t[i][tInd][2]) for i in 1:nEvalFine];
    # RHOgrid_MOC = [(XU_t[i,j][tInd][3]) for i in 1:nEvalFine, j in 1:nEvalFine]; # no change in variabless
    RHOgrid_NN = [ρFn([XU_t[i][tInd][1], tVal]) for i in 1:nEvalFine];

    figure(45, (8,4)); clf();
    subplot(1,2,1);
    plot(X1grid, RHOgrid_MOC, label = "MOC"); 
    xlabel("x1"); ylabel("x2");
    xlim(-maxval, maxval);
    plot(X1grid, RHOgrid_NN, label = "NN"); 
    legend();

    subplot(1,2,2);
    plot(X1grid, abs.(RHOgrid_MOC - RHOgrid_NN)); 
    ϵ_mse = sum(abs2, (RHOgrid_MOC - RHOgrid_NN))/(nEvalFine^2);
    ϵ_mse_str = string(@sprintf "%.2e" ϵ_mse);
    xlabel("x1"); ylabel("x2");
    xlim(-maxval, maxval);
    title("Pointwise ϵ");
    t_str = string(@sprintf "%.2f" tVal);
    suptitle("t = $(t_str)")
    tight_layout();

    savefig("figs_moc/exp$(expNum)/t$(tInd).png");
    sleep(0.5);
end