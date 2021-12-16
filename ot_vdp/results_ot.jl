## Plot the results of the OT-PINNs implementation for the Van der Pol oscillator
using JLD2,
    PyPlot, NeuralPDE, ModelingToolkit, LinearAlgebra, Flux, ForwardDiff, Trapz, Printf
pygui(true);

# Load data 
activFunc = tanh;
suff = string(activFunc);
nn = 48;
otIters = 20;
maxNewPts = 200;

cd(@__DIR__);
# fileLoc = "data/dx1eM1_ot1Eval_vdp_$(suff)_$(nn)_ot$(otIters)_mnp$(maxNewPts)_gpu_otShStab.jld2";
fileLoc = "data/dx1eM1_ot1Eval_vdp_tanh_48_ot20_mnp200.jld2";

println("Loading file");
file = jldopen(fileLoc, "r");
optParams = read(file, "optParams");
PDE_losses = read(file, "PDE_losses");
newPtsAll = read(file, "newPtsAll");
BC_losses = read(file, "BC_losses");
pde_train_sets = read(file, "pde_train_sets");
close(file);

nEvalFine = 100;

# Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
parameterless_type_θ = DiffEqBase.parameterless_type(optParams[1]);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

# Van der Pol Dynamics
f(x) = [x[2]; -x[1] + (1 - x[1]^2) * x[2]];

Q = 0.1; # Q = σ^2

# Domain
maxval = 4.0;

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

# loop over all OT iterations, get errors
for otIter = 0:otIters-1
    otIter += 1
    @info "Analysing after $(otIter-1) OT iterations:"
    # Generate functions to check solution and PDE error
    function ρ_pdeErr_fns(optParam)
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
    ρFn, pdeErrFn = ρ_pdeErr_fns(optParams[otIter])

    ## new points discovered from OT
    if otIter < otIters + 1
        newPts = newPtsAll[otIter]
    end

    function compute_RHO_PDEerr(nEvalFine, ρFn, pdeErrFn)
        xxFine = range(-maxval, maxval, length = nEvalFine)
        yyFine = range(-maxval, maxval, length = nEvalFine)

        RHOFine = [ρFn([x, y]) for x in xxFine, y in yyFine]
        # normalize
        RHOFine = RHOFine / trapz((xxFine, yyFine), RHOFine)

        pdeErrFine = [pdeErrFn([x, y])^2 for x in xxFine, y in yyFine]


        return RHOFine, pdeErrFine
    end
    RHOFineVec[otIter], pdeErrFineVec[otIter] = compute_RHO_PDEerr(nEvalFine, ρFn, pdeErrFn)

    function get_mseEqErr(pdeErrFine)
        return sum(pdeErrFine[:] .^ 2) / length(pdeErrFine)
    end
    mseEqErrVec[otIter] = get_mseEqErr(pdeErrFineVec[otIter])
    println(
        "ϵ_pde = $(mseEqErrVec[otIter]), ϵ_ptMax = $(maximum(pdeErrFineVec[otIter])) ",
    )
end

## Plotting on evaluation grid
otIter = 10;
otIter += 1; # Check for 10th iteration (plot shown in paper)
println("Plotting on evaluation set");
function plotDistErr(RHOFine, pdeErrFine, figNum)
    mseEqErrStr = @sprintf "%.2e" mseEqErrVec[otIter]
    figure(figNum, [8, 4])
    clf()
    subplot(1, 2, 1)
    pcolormesh(XXFine, YYFine, RHOFine, shading = "auto", cmap = "inferno")
    colorbar()
    title("Steady-State Solution (ρ)")
    xlabel("x1")
    ylabel("x2")
    axis("auto")
    tight_layout()

    subplot(1, 2, 2)
    pcolor(XXFine, YYFine, pdeErrFine, shading = "auto", cmap = "inferno")
    colorbar()
    if otIter < otIters + 1
        newPts = newPtsAll[otIter]
        scatter(newPts[1, :], newPts[2, :], s = 1.0, color = "w")
    end

    title(L"Equation Error; $ϵ_{pde}$ = %$(mseEqErrStr)")
    xlabel("x1")
    ylabel("x2")
    axis("auto")
    tight_layout()
end
plotDistErr(RHOFineVec[otIter], pdeErrFineVec[otIter], otIter)
# savefig("figs_prelim/otSoln_vdp.png");

## Plot eqErr vs. OT
figure(1000);
clf();
semilogy(1:otIters, mseEqErrVec);
scatter(1:otIters, mseEqErrVec);
xlabel("OT iterations");
xticks(1:otIters);
ylabel("ϵ");
title(L"$ϵ_{pde}$");
nNN0 = size(pde_train_sets[1][1], 2);
tight_layout();
# savefig("figs_prelim/otError_vdp.png");

## Plot maxPtWiseErr after OT
figure(1001); clf();
semilogy(1:otIters, maximum.(pdeErrFineVec));
scatter(1:otIters, maximum.(pdeErrFineVec));
xlabel("OT iterations");
xticks(1:otIters);
ylabel("ϵ");
title("Maximum Pointwise ϵ");
tight_layout();