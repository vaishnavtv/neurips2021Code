## Plot the results of the OT-PINNs implementation for the Van der Pol Rayleigh oscillator
using JLD2,
    PyPlot,
    NeuralPDE,
    ModelingToolkit,
    LinearAlgebra,
    Flux,
    ForwardDiff,
    Trapz,
    Printf,
    LaTeXStrings
pygui(true);

# Load data 
activFunc = tanh; #dx = 0.25;
suff = string(activFunc);
nn = 48;
otIters = 20;
maxNewPts = 200;

cd(@__DIR__);
fileLoc = "data/dx25eM2_ot1Eval_vdpr_$(suff)_$(nn)_ot$(otIters)_mnp$(maxNewPts)_otEmd.jld2";

println("Loading file");
file = jldopen(fileLoc, "r");
optParams = read(file, "optParams");
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
pde_train_sets = read(file, "pde_train_sets");
newPtsAll = read(file, "newPtsAll");
close("file");

nEvalFine = 100;
Q = 0.3;
ηTrue(x) = 1 / Q * (x[1]^2 + x[2]^2 - 1 / 2 * (x[1]^2 + x[2]^2)^2);
rhoTrue(x) = exp(ηTrue(x));

# Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
parameterless_type_θ = DiffEqBase.parameterless_type(optParams[1]);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);


# Van der Pol Rayleigh Dynamics
f(x) = [x[2]; -x[1] + (1 - x[1]^2 - x[2]^2) * x[2]];

maxval = 2.0;

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

# loop over all OT iterations
for otIter = 0:otIters-1
    otIter += 1
    println("Analysing after $(otIter-1) OT iterations:")
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
        "The mean squared equation error after $(otIter-1) iterations is: $(mseEqErrVec[otIter])",
    )
end

## Plotting on fine grid
otIter = 10;
otIter += 1;
println("Plotting on evaluation set");
function plotDistErr(nEvalFine, RHOFine, pdeErrFine, figNum)
    xxFine = range(-maxval, maxval, length = nEvalFine)
    yyFine = range(-maxval, maxval, length = nEvalFine)

    RHOTrue = [rhoTrue([x, y]) for x in xxFine, y in yyFine]

    # normalize
    RHOTrue = RHOTrue / trapz((xxFine, yyFine), RHOTrue)
    RHOErr = abs.(RHOFine - RHOTrue)

    println(
        "The mean squared absolute error in the solution after $(otIter-1) iterations is:",
    )
    mseRHOErr = sum(RHOErr[:] .^ 2) / length(RHOErr)
    @show mseRHOErr
    mseRHOErrStr = @sprintf "%.2e" mseRHOErr
    mseEqErrStr = @sprintf "%.2e" mseEqErrVec[otIter]

    figure(figNum, (12, 4))
    clf()
    subplot(1, 3, 1)
    pcolormesh(XXFine, YYFine, RHOFine, cmap = "inferno", shading = "auto")
    colorbar()
    xlabel("x1")
    ylabel("x2")
    title("Prediction")
    axis("auto")
    tight_layout()

    subplot(1, 3, 2)
    pcolormesh(XXFine, YYFine, RHOTrue, cmap = "inferno", shading = "auto")
    colorbar()
    xlabel("x1")
    ylabel("x2")
    title("Exact")
    axis("auto")
    tight_layout()

    subplot(1, 3, 3)
    pcolormesh(XXFine, YYFine, RHOErr, shading = "auto", cmap = "inferno")
    colorbar()
    title(L"Solution Error; $ϵ_{ρ}=$ %$mseRHOErrStr")
    xlabel("x1")
    ylabel("x2")
    axis("auto")
    tight_layout()

end
plotDistErr(nEvalFine, RHOFineVec[otIter], pdeErrFineVec[otIter], otIter)

## Plot eqErr vs. OT
println("Plotting equation error vs. OT")
figure(1000);
clf();
semilogy(1:otIters, mseEqErrVec);
scatter(1:otIters, mseEqErrVec);
xlabel("OT iterations");
xticks(1:otIters);
ylabel("ϵ");
title(L"$ϵ_{pde}$");
tight_layout();

