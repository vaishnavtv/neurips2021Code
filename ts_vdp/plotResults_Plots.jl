## Plot the results of the baseline-PINNs implementation for transient vdp
using JLD2,
    NeuralPDE,
    ModelingToolkit,
    LinearAlgebra,
    Flux,
    Trapz,
    Printf,
    LaTeXStrings,
    ForwardDiff,
    Plots
Plots.pyplot();

# Van der Pol Dynamics
f(x) = [x[2]; -x[1] + (1-x[1]^2)*x[2]];
df(x) = ForwardDiff.jacobian(f, x)

function g(x::Vector)
    return [0.0f0;1.0f0];
end
# Load data 
activFunc = tanh;
dx = 0.01;
suff = string(activFunc);
nn = 48;
expNum = 15;
tEnd = 10.0
strategy = "grid";
# chain = Chain(Dense(3, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain = Chain(Dense(3, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));

Q_fpke = 0.1f0#*1.0I(2); # σ^2
# diffC = 0.5 * (g(xSym) * Q_fpke * g(xSym)'); # diffusion coefficient (constant in our case, not a fn of x)
diffCTerm(x) = 0.5 * (g(x) * Q_fpke * g(x)'); 

cd(@__DIR__);
fileLoc = "dataTS_$(strategy)/ll_ts_vdp_exp$(expNum).jld2";

println("Loading file from exp $(expNum)");
file = jldopen(fileLoc, "r");
optParam = read(file, "optParam");
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
close(file);
println("Are any of the parameters NaN? $(any(isnan.(optParam)))")
## plot losses
nIters = length(PDE_losses);
# figure(1); clf();
# semilogy(1:nIters, PDE_losses, label =  "PDE");
# semilogy(1:nIters, BC_losses, label = "BC");
# xlabel("Iterations");
# ylabel("ϵ");
# title(string(strategy," Exp $(expNum)"));
# legend();
# tight_layout();
## Neural network
parameterless_type_θ = DiffEqBase.parameterless_type(optParam);
phi = NeuralPDE.get_phi(chain, parameterless_type_θ);

# Generate functions to check solution and PDE error
function ρ_pdeErr_fns(optParam)
    ρNetS(xt) = exp(first(phi(xt, optParam)) )
    dρ(xt) = ForwardDiff.gradient(ρNetS,xt);
    dρ_x(xt) =  dρ(xt)[1:2]; dρ_t(xt) = dρ(xt)[end];
    d2ρ_xx(xt) = ForwardDiff.hessian(ρNetS, xt)[1:2,1:2];
    
    function pdeErrFn(xt)
        x = xt[1:2]; t = xt[end];
        out = (dρ_t(xt) + dot(f(x), dρ_x(xt)) + sum(ρNetS(xt).*df(x)) - sum(diffCTerm(x).*d2ρ_xx(xt)))#/(ρNetS(xt))
        return out
    end

    function pdeErrFn_tEnd(xt)
        x = xt[1:2]; t = xt[end];
        out = (dot(f(x), dρ_x(xt)) + sum(ρNetS(xt).*df(x)) - sum(diffCTerm(x).*d2ρ_xx(xt)))#/(ρNetS(xt))
        return out
    end

    return ρNetS, pdeErrFn, pdeErrFn_tEnd
end
ρFn, pdeErrFn, pdeErrFn_tEnd = ρ_pdeErr_fns(optParam);
# xt0 = [-1,1,5]; x0 = xt0[1:2]; t0 = xt0[end];
# ρNetS(xt0)
# dρ(xt0)
# dρ_x(xt0)
# dρ_t(xt0)
# d2ρ_xx(xt0)
# dot(f(x0), dρ_x(xt0))
# sum(ρNetS(xt0).*df(x0))
# pdeErrFn(xt0)
# Domain
nEvalFine = 100;
maxval = 4.0f0;
t0 = 0.0; 
xxFine = range(-maxval, maxval, length = nEvalFine);
yyFine = range(-maxval, maxval, length = nEvalFine);
ttFine = range(t0, tEnd, length = 10);

RHOPred = [ρFn([x, y, t]) for x in xxFine, y in yyFine, t in ttFine];
pdeErrFine = [pdeErrFn([x, y, t])^2 for x in xxFine, y in yyFine, t in ttFine]; # squared equation error on the evaluation grid
pdeErrFine_tEnd = [pdeErrFn_tEnd([x, y, tEnd])^2 for x in xxFine, y in yyFine]; # squared equation error on the evaluation grid
@show maximum(RHOPred)

# normalize
# normC = trapz((xxFine, yyFine), RHOPred)
# RHOFine = RHOPred / normC;

# initial condition error
mseICErr = sum(RHOPred[:,:,1] .- 1f-5)/(nEvalFine^2);
@show mseICErr;

println("The mean squared equation error with dx=$(dx) is:")
pdeErr2End = [sum(pdeErrFine[:,:,i]) for i in 2:length(ttFine)]; # from t=t1 (step after t0)
mseEqErr = sum(pdeErr2End)/ (length(pdeErrFine) - nEvalFine^2);
# mseEqErr = sum(pdeErrFine[:] .^ 2) / length(pdeErrFine);
@show mseEqErr;
mseEqErrStr = @sprintf "%.2e" mseEqErr;
XXFine = zeros(nEvalFine, nEvalFine)
YYFine = similar(XXFine)

for i = 1:nEvalFine, j = 1:nEvalFine
    XXFine[i, j] = xxFine[i]
    YYFine[i, j] = yyFine[j]
end

## Plot shown in paper
function plotDistErr(figNum)
    # tInd = 3
    for (tInd,tVal) in enumerate(ttFine)
        figure(figNum, [8, 4])
        clf()
        subplot(1, 2, 1)
        # figure(figNum); clf();
        pcolor(XXFine, YYFine, RHOPred[:,:,tInd], shading = "auto", cmap = "inferno")
        colorbar()
        xlabel("x1")
        ylabel("x2")
        axis("auto")
        title("Steady-State Solution (ρ)")

        subplot(1, 2, 2)
        if tInd != length(ttFine)
            pcolor(XXFine, YYFine, pdeErrFine[:,:,tInd], cmap = "inferno", shading = "auto")
            title("Pointwise PDE Error")
        else
            pcolor(XXFine, YYFine, pdeErrFine_tEnd, cmap = "inferno", shading = "auto")
            title("Pointwise PDE Error (Steady-state)")
        end
        colorbar()
        axis("auto")
        # title("Pointwise PDE Error")
        # title(L"Equation Error; $ϵ_{pde}$ = %$(mseEqErrStr)")
        xlabel("x1")
        ylabel("x2")
        suptitle("t = $(tVal)")
        tight_layout()
        sleep(0.2);
    end
end
# plotDistErr(expNum);
##

# p1 = Plots.plot(xxFine, yyFine, RHOPred[:,:,1]', linetype=:heatmap)
# # title!("Steady-state Solution (ρ)")
# xlabel!("x1"); ylabel!("x2");
# plot(p1)

##
# i = 10;
#         p1 = Plots.plot(xxFine, yyFine, RHOPred[:,:,i]',st=:heatmap, title="Steady-State Solution (ρ)")
#         xlabel!("x1"); ylabel!("x2");
#         title = @sprintf("error")
#         p2 = Plots.plot(xxFine, yyFine, pdeErrFine[:,:,i]',st=:heatmap, title = "Pointwise PDE Error")
#         xlabel!("x1"); ylabel!("x2");
#         Plots.plot(p1,p2, size = (800,400), plot_title = "t = ")

## plotting in plots.
function plot_(res)
    # Animate
    anim = @animate for (i, t) in enumerate(ttFine)
        @info "Animating frame $i..."
        suptitle = @sprintf("t = %.3f", t)
        p1 = Plots.plot(xxFine, yyFine, RHOPred[:,:,i]',st=:heatmap, title="Steady-State Solution (ρ)")
        xlabel!("x1"); ylabel!("x2");
        p2 = Plots.plot(xxFine, yyFine, pdeErrFine[:,:,i]',st=:heatmap, title = "Pointwise PDE Error")
        xlabel!("x1"); ylabel!("x2");
        Plots.plot(p1,p2, size = (800,400), plot_title = suptitle)
    end
    gif(anim,"figs/ts_vdp_exp$(expNum).gif", fps=1)
end
plot_(1)
# savefig("figs/quasi_exp$(expNum).png");
##

# ## normalisation as quadrature problem  
# using Quadrature
# ρFnQuad(z,p) = ρFn(z)#/normC
# prob = QuadratureProblem(ρFnQuad, [minM, minα], [maxM, maxα], p = 0);
# sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
# @show sol.u
# @show abs(sol.u - normC)

