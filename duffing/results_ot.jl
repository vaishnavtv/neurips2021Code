## Plot the results of the OT-PINNs implementation for the Duffing oscillator
using JLD2,
    PyPlot,
    NeuralPDE,
    ModelingToolkit,
    LinearAlgebra,
    Flux,
    ForwardDiff,
    Trapz,
    Printf,
    LaTeXStrings,
    DiffEqFlux
pygui(true);
using Statistics
# Load data 
activFunc = tanh; #dx = 0.25;
suff = string(activFunc);
nn = 48;
otIters = 20;
maxNewPts = 200;
strat = "ot";
expNum = 5;

cd(@__DIR__);
fileLoc = "data_$(strat)/$(strat)_duff_exp$(expNum).jld2";

println("Loading file");
file = jldopen(fileLoc, "r");
optParams = read(file, "optParams");
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
pde_train_sets = read(file, "pde_train_sets");
newPtsAll = read(file, "newPtsAll");
close("file");

nEvalFine = 100;
Q_fpke = 0.1;
## True solution
rhoTrue(x) = exp(-η_duff / Q_fpke * (x[2].^2 + α_duff.*x[1].^2 + β_duff/2*x[1].^4));

# Duffing oscillator Dynamics
η_duff = 0.2; α_duff = 1.0; β_duff = 0.2;
f(x) = [x[2]; η_duff.*x[2] .- α_duff.*x[1] .- β_duff.*x[1].^3];

function g(x::Vector)
    return [0.0f0;1.0f0];
end # diffusion

maxval = 2.0;

# Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
parameterless_type_θ = DiffEqBase.parameterless_type(optParams[1]);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

## set up error function
@parameters x1, x2
@variables η(..)

xSym = [x1; x2]
# PDE
ρ(x) = exp(η(x...));
#  PDE written directly in η
diffC = 0.5f0*(g(xSym)*Q_fpke*g(xSym)'); # diffusion term
G = diffC*η(xSym...);

T1 = sum([(Differential(xSym[i])(f(xSym)[i]) + (f(xSym)[i]* Differential(xSym[i])(η(xSym...)))) for i in 1:length(xSym)]); # drift term
T2 = sum([(Differential(xSym[i])*Differential(xSym[j]))(G[i,j]) for i in 1:length(xSym), j=1:length(xSym)]);
T2 += sum([(Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j]) - (Differential(xSym[i])*Differential(xSym[j]))(diffC[i,j])*η(xSym...) + diffC[i,j]*(Differential(xSym[i])(η(xSym...)))*(Differential(xSym[j])(η(xSym...))) for i in 1:length(xSym), j=1:length(xSym)]); # complete diffusion term

Eqn = expand_derivatives(-T1+T2); 
pdeOrig = simplify(Eqn, expand = true) ~ 0.0f0;
pde = pdeOrig;

initθ = DiffEqFlux.initial_params(chain) #|> gpu;
strategy = NeuralPDE.GridTraining(0.1);
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



## grid for plotting
function gridXY(nEvalFine)
    xxFine = range(-maxval, maxval, length = nEvalFine)
    yyFine = range(-maxval, maxval, length = nEvalFine)
    XXFine = zeros(nEvalFine, nEvalFine)
    YYFine = similar(XXFine)
    for i = 1:nEvalFine, j = 1:nEvalFine
        XXFine[i, j] = xxFine[i]
        YYFine[i, j] = yyFine[j]
    end
    return XXFine, YYFine
end
XXFine, YYFine = gridXY(nEvalFine);

## initialize loop
mseEqErrVec = Vector{Float64}(undef, otIters);
RHOFineVec = Vector{Matrix{Float64}}(undef, otIters);
pdeErrFineVec = Vector{Matrix{Float64}}(undef, otIters);
RHOErrFineVec = Vector{Matrix{Float64}}(undef, otIters);
mseRHOErrVec = Vector{Float64}(undef, otIters);
maxRHOErrVec = similar(mseRHOErrVec);

# loop over all OT iterations>
xxFine = range(-maxval, maxval, length = nEvalFine)
yyFine = range(-maxval, maxval, length = nEvalFine)
RHOTrue = [rhoTrue([x, y]) for x in xxFine, y in yyFine]
# normalize
RHOTrue = RHOTrue / trapz((xxFine, yyFine), RHOTrue);
for otIter = 0:otIters-1

    otIter += 1
    @info "Analysing after $(otIter-1) OT iterations:"
    function ρ_pdeErr_fns(optParam)
        function ηNetS(x)
            return first(phi(x, optParam))
        end
        ρNetS(x) = exp(ηNetS(x)) # solution after first iteration
        pdeErrFn(x) = first(_pde_loss_function(x, optParam))
        
        return ρNetS, pdeErrFn

    end
    ρFn, pdeErrFn = ρ_pdeErr_fns(optParams[otIter])

    ## new points discovered from OT
    if otIter < otIters + 1
        newPts = newPtsAll[otIter]
    end

    function compute_RHO_PDEerr(nEvalFine, ρFn, pdeErrFn)
       
        RHOFine = [ρFn([x, y]) for x in xxFine, y in yyFine]
        # normalize
        RHOFine = RHOFine / trapz((xxFine, yyFine), RHOFine)

        pdeErrFine = [pdeErrFn([x, y])^2 for x in xxFine, y in yyFine]

        RHOErrFine = abs.(RHOFine - RHOTrue)
        
        return RHOFine, pdeErrFine, RHOErrFine
    end
    RHOFineVec[otIter], pdeErrFineVec[otIter], RHOErrFineVec[otIter] = compute_RHO_PDEerr(nEvalFine, ρFn, pdeErrFn)

    function get_mse(ErrFine)
        return mean(abs2, ErrFine) # sum(pdeErrFine[:] .^ 2) / length(pdeErrFine)
    end
    mseEqErrVec[otIter] = get_mse(pdeErrFineVec[otIter])
    mseRHOErrVec[otIter] =  get_mse(RHOErrFineVec[otIter])
    maxRHOErrVec[otIter] = maximum(RHOErrFineVec[otIter]);
    println(
        "ϵ_pde = $(mseEqErrVec[otIter])",
    )
end

## Plotting on fine grid
otIter = 4;
otIter += 1;
println("Plotting on evaluation set");
function plotDistErr(nEvalFine, RHOFine, pdeErrFine, figNum)
    # xxFine = range(-maxval, maxval, length = nEvalFine)
    # yyFine = range(-maxval, maxval, length = nEvalFine)

    # RHOTrue = [rhoTrue([x, y]) for x in xxFine, y in yyFine]

    # # normalize
    # RHOTrue = RHOTrue / trapz((xxFine, yyFine), RHOTrue)
    # RHOErr = abs.(RHOFine - RHOTrue)

    println(
        "The mean squared absolute error in the solution after $(otIter-1) iterations is:",
    )
    # mseRHOErr = mean(abs2, RHOErr);  #sum(RHOErr[:] .^ 2) / length(RHOErr)
    @show mseRHOErrVec[otIter]
    mseRHOErrStr = @sprintf "%.2e" mseRHOErrVec[otIter]
    mseEqErrStr = @sprintf "%.2e" mseEqErrVec[otIter]

    figure(figNum, (8,8))
    clf()
    subplot(2, 2, 1)
    pcolor(XXFine, YYFine, RHOFine, cmap = "inferno", shading = "auto")
    colorbar()
    
    xlabel("x1")
    ylabel("x2")
    title("Prediction")
    axis("auto")
    tight_layout()

    subplot(2, 2, 2)
    pcolor(XXFine, YYFine, RHOTrue, cmap = "inferno", shading = "auto")
    colorbar()
    xlabel("x1")
    ylabel("x2")
    title("Exact")
    axis("auto")
    tight_layout()

    subplot(2, 2, 3)
    pcolor(XXFine, YYFine, pdeErrFine, shading = "auto", cmap = "jet")
    colorbar()
    if otIter < otIters + 1
        newPts = newPtsAll[otIter]
        scatter(newPts[1, :], newPts[2, :], s = 0.1, color = "w")
    end
    title("ϵ_pde = $(mseEqErrStr)");
    # title(L"Solution Error; $ϵ_{ρ}=$ %$mseRHOErrStr")
    xlabel("x1")
    ylabel("x2")
    axis("auto")
    tight_layout()

    subplot(2, 2, 4)
    pcolor(XXFine, YYFine, RHOErrFineVec[otIter], shading = "auto", cmap = "jet")
    colorbar()
    title("ϵ_ρ = $(mseRHOErrStr)");
    xlabel("x1")
    ylabel("x2")
    axis("auto")
    tight_layout()

    suptitle("After $(otIter-1) iterations");
    tight_layout();

end
plotDistErr(nEvalFine, RHOFineVec[otIter], pdeErrFineVec[otIter], otIter)
# savefig("figs_ot/exp$(expNum)_otIter$(otIter).png");

## Plot eqErr vs. OT
println("Plotting equation error vs. OT")
figure(1000);
clf();
semilogy(1:otIters, mseEqErrVec);
scatter(1:otIters, mseEqErrVec);
xlabel("OT iterations");
xticks(1:otIters);
ylabel("ϵ");
title(L"$ϵ_{\rm pde}$");
tight_layout();
# savefig("figs_prelim/otError_vdpr.png");


