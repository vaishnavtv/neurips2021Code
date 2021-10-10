## Plot the results of the baseline-PINNs implementation for transient vdp
using JLD2,
    PyPlot,
    NeuralPDE,
    ModelingToolkit,
    LinearAlgebra,
    Flux,
    Trapz,
    Printf,
    LaTeXStrings,
    ForwardDiff
pygui(true);
# @variables x1, x2, t
# xSym = [x1, x2];

# Van der Pol Dynamics
f(x) = -1.0f0*x;
df(x) = ForwardDiff.jacobian(f, x)

function g(x::Vector)
    return [0.0f0;1.0f0];
end
# Load data 
activFunc = tanh;
dx = 0.01;
suff = string(activFunc);
nn = 48;
expNum = 6;
@info "Plotting results for ts_linsys experiment number $(expNum)"
simMOC = true;
strategy = "grid";
# chain = Chain(Dense(3, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));
chain = Chain(Dense(3, nn, activFunc), Dense(nn, nn, activFunc), Dense(nn, 1));

t0 = 0.0; tEnd = 1.0f0;
Q_fpke = 0.0f0#*1.0I(2); # σ^2
Q_fpke_str = string(Q_fpke);

# diffC = 0.5 * (g(xSym) * Q_fpke * g(xSym)'); # diffusion coefficient (constant in our case, not a fn of x)
diffCTerm(x) = 0.5 * (g(x) * Q_fpke * g(x)'); 

cd(@__DIR__);
fileLoc = "dataTS_$(strategy)/ll_ts_ls_exp$(expNum).jld2";

println("Loading file from exp $(expNum)");
file = jldopen(fileLoc, "r");
optParam = read(file, "optParam");
PDE_losses = read(file, "PDE_losses");
BC_losses = read(file, "BC_losses");
# IC_losses = read(file, "IC_losses");
close(file);
println("Are any of the parameters NaN? $(any(isnan.(optParam)))")
## plot losses
nIters = length(PDE_losses);
figure(1); clf();
semilogy(1:nIters, PDE_losses, label =  "PDE");
semilogy(1:nIters, BC_losses, label = "BC");
# semilogy(1:nIters, IC_losses, label = "IC");
xlabel("Iterations");
ylabel("ϵ");
title(string(strategy," Exp $(expNum)"));
legend();
tight_layout();
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
nEvalFine = 100; # pts along x,y
ntFine = 10; # pts along t
maxval = 4.0f0;

xxFine = range(-maxval, maxval, length = nEvalFine);
yyFine = range(-maxval, maxval, length = nEvalFine);
ttFine = range(t0, tEnd, length = ntFine);
nEvalFine = length(xxFine);
RHOPred = [ρFn([x, y, t]) for x in xxFine, y in yyFine, t in ttFine];
pdeErrFine = [pdeErrFn([x, y, t])^2 for x in xxFine, y in yyFine, t in ttFine]; # squared equation error on the evaluation grid
pdeErrFine_tEnd = [pdeErrFn_tEnd([x, y, tEnd])^2 for x in xxFine, y in yyFine]; # squared equation error on the evaluation grid
@show maximum(RHOPred)

# normalize
RHOFine = deepcopy(RHOPred);
for i in 1:length(ttFine)
    normC = trapz((xxFine, yyFine), RHOPred[:,:,i])
    RHOFine[:,:,i] = RHOPred[:,:,i] / normC;
end

# initial condition error
μ_ss = [0.0f0,0.0f0]; 
Σ_ss = 0.1f0*[1.0f0; 1.0f0] .* 1.0f0I(2);
inv_Σ_ss = Float32.(inv(Σ_ss))
sqrt_det_Σ_ss = Float32(sqrt(det(Σ_ss)));

ρ0(x) =  exp(-0.5f0 * (x - μ_ss)' * inv_Σ_ss * (x - μ_ss)) / (2.0f0 * Float32(pi) * sqrt_det_Σ_ss); # ρ at t0, Gaussian
RHO0TRUE = [ρ0([x,y]) for x in xxFine, y in yyFine];
# mseICErr = sum(abs2, RHOPred[:,:,1] .- 0.00015625f0);#/(nEvalFine^2);
mseICErr = sum(abs2, RHOPred[:,:,1] - RHO0TRUE)/(nEvalFine^2);
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
    # tInd = 10; tVal = ttFine[tInd]
        figure(figNum, [8, 4])
        clf()
        subplot(1, 2, 1)
        # figure(figNum); clf();
        pcolor(XXFine, YYFine, RHOPred[:,:,tInd], shading = "auto", cmap = "inferno")
        colorbar()
        xlabel("x1")
        ylabel("x2")
        axis("auto")
        title("PDF (ρ)")

        subplot(1, 2, 2)
        # if tInd != length(ttFine)
            pcolor(XXFine, YYFine, pdeErrFine[:,:,tInd], cmap = "inferno", shading = "auto")
            title("Pointwise PDE Error")
        # else
        #     pcolor(XXFine, YYFine, pdeErrFine_tEnd, cmap = "inferno", shading = "auto")
        #     title("Pointwise PDE Error (Steady-state)")
        # end
        colorbar()
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
plotDistErr(expNum);


## compare against MOC
mkpath("figs/exp$(expNum)") # to save figs
using DifferentialEquations 
uDyn(rho, x) = -tr(df(x)); 
# uDyn(rho,x) = -rho*tr(df(x));
tInt = ttFine[2] - ttFine[1];
function nlSimContMOC(x0)
    odeFn(xu,p,t) = [f(xu[1:2]); uDyn(xu[end], xu[1:2])]
    prob = ODEProblem(odeFn, x0, (0.0, tEnd));
    sol = solve(prob, Tsit5(), saveat = tInt, reltol = 1e-3, abstol = 1e-3);
    return sol.u
end
RHO0_NN = RHOFine[:,:,1];
normC0 = trapz((xxFine, yyFine), RHOPred[:,:,1]); # normalisation for initial state
if simMOC
    minval = -maxval; 
    XU_t = [nlSimContMOC([x, y, 0.0f0]) for x in xxFine, y in yyFine];
    # XU_t = [nlSimContMOC([x, y, ρFn([x, y, 0.0f0])/normC0 ]) for x in xxFine, y in yyFine];
    
    # tSpan = 0:0.01:0.1;
    ##
    for (tInd, tVal) in enumerate(ttFine)
    # tInd = 1; tVal = tSpan[tInd];
        X1grid = [XU_t[i,j][tInd][1] for i in 1:nEvalFine, j in 1:nEvalFine];
        X2grid = [XU_t[i,j][tInd][2] for i in 1:nEvalFine, j in 1:nEvalFine];
        RHOgrid_MOC = [RHO0_NN[i,j]*exp(XU_t[i,j][tInd][3]) for i in 1:nEvalFine, j in 1:nEvalFine];
        # RHOgrid_MOC = [(XU_t[i,j][tInd][3]) for i in 1:nEvalFine, j in 1:nEvalFine]; # no change in variabless
        normC_moc = trapz((X1grid[:,1], X2grid[2,:]), RHOgrid_MOC)
        @show normC_moc;
        x1Grid = X1grid[:,1]; x2Grid = X2grid[2,:];
        RHOgrid_NN = [ρFn(XU_t[i,j][tInd]) for i in 1:nEvalFine, j in 1:nEvalFine];
        # RHOgrid_NN = [ρFn([x, y, tVal]) for x in x1Grid, y in x2Grid];
        normC_nn = trapz((X1grid[:,1], X2grid[2,:]), RHOgrid_NN);
        RHOgrid_NN /= normC_nn;
        @show normC_nn;

        figure(45, (12,4)); clf();
        subplot(1,3,1);
        contourf(X1grid, X2grid, RHOgrid_MOC); colorbar();
        xlabel("x1"); ylabel("x2");
        xlim(minval, maxval);
        ylim(minval,  maxval);
        title("MOC");

        subplot(1,3,2);
        contourf(X1grid, X2grid, RHOgrid_NN); colorbar();
        xlabel("x1"); ylabel("x2");
        xlim(minval, maxval);
        ylim(minval,  maxval);
        title(" FPKE_NN (Q = $(Q_fpke_str))");

        subplot(1,3,3);
        contourf(X1grid, X2grid, abs.(RHOgrid_MOC - RHOgrid_NN)); 
        ϵ_mse = sum(abs2, (RHOgrid_MOC - RHOgrid_NN))/(nEvalFine^2);
        ϵ_mse_str = string(@sprintf "%.2e" ϵ_mse);
        colorbar();
        xlabel("x1"); ylabel("x2");
        xlim(minval, maxval);
        ylim(minval,  maxval);
        title("Pointwise ϵ");
        t_str = string(@sprintf "%.2f" tVal);
        suptitle("t = $(t_str)")
        tight_layout();

        savefig("figs/exp$(expNum)/t$(tInd).png");
        # sleep(0.1);
    end
end
##

## normalisation as quadrature problem  
# using Quadrature
# ρFnQuad(z,p) = ρFn(z)#/normC
# prob = QuadratureProblem(ρFnQuad, [minM, minα], [maxM, maxα], p = 0);
# sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
# @show sol.u
# @show abs(sol.u - normC)

