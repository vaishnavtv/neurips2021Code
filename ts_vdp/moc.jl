# solving missile fpke using MOC
# Pg. 25 from Andrea Weiss's thesis
using LinearAlgebra, DifferentialEquations, ForwardDiff, Trapz, Printf
using PyPlot; pygui(true);
cd(@__DIR__);
simMOC = false;
simMOC_c = false;
simDyn = false;
tEnd = 1.0; tInt = tEnd/10;
propRHO = true; # propagate ρ̇ = - ρ*(∇.f(x)) if true, else propagate η̇ = -tr(∇.f(x)), where η = ln(ρ/ρ0) and η0 = 0.

# van der pol oscillator
# f(x) = [x[2]; (-x[1] + (1 - x[1]^2) * x[2])]; 
# minval = -4.0f0; maxval = 4.0f0; 
# μ_0 = [0,0]; 
# Σ_0 = 0.1*[1; 1] .* 1.0I(2);

## michaelis-menten model from Andrea Weiss' thesis
Km = 1;
f(x) = [-x[2]/(Km + x[1])*x[1]; 0]; 
minval = 0.0f0; maxval = 3.0f0; # for MM Model, use tEnd = 1.0
μ_0 = [2,2]; # for mm model
Σ_0 = [1/8; 1/40] .* 1.0I(2); # for mm model

# f(x) = -1.0f0.*x # linear dynamcis

df(x) = ForwardDiff.jacobian(f,x);

if propRHO
    uDyn(rho,x) = -rho*tr(df(x)); # propagating rho directly
else
    uDyn(rho, x) = -tr(df(x)); # propagating log(rho/rho0)
end

# propagate density using MOC
# if simMOC 
function nlSimContMOC(x0)
    odeFn(xu,p,t) = [f(xu[1:2]); uDyn(xu[end], xu[1:2])]
    prob = ODEProblem(odeFn, x0, (0.0, tEnd));
    sol = solve(prob, Tsit5(), saveat = tInt, reltol = 1e-3, abstol = 1e-3);
    return sol.u
end
nEvalFine = 100; # initial grid, no. of points along each axis
nGridPnts = nEvalFine^2;

xxFine = range(minval, maxval, length = nEvalFine);
yyFine = range(minval, maxval, length = nEvalFine);
XXFine = zeros(nEvalFine, nEvalFine); YYFine = similar(XXFine);
for i = 1:nEvalFine, j = 1:nEvalFine
    XXFine[i, j] = xxFine[i]
    YYFine[i, j] = yyFine[j]
end

# XU0 = [[x, y, 1/nGridPnts] for x in xxFine, y in yyFine]; 

rho0(x) =  exp(-1 / 2 * (x - μ_0)' * inv(Σ_0) * (x - μ_0)) / (2 * pi * sqrt(det(Σ_0))); # ρ at t0
# rho0(x) = 1/64;
RHO0 = [rho0([x,y]) for x in xxFine, y in yyFine ];
norm_RHO0 = trapz((xxFine, yyFine), RHO0);
@show norm_RHO0
# XU0 =  [[x, y, rho0([x,y])] for x in xxFine, y in yyFine];
##
if propRHO
    XU_t = [nlSimContMOC([x, y, rho0([x,y])]) for x in xxFine, y in yyFine];
else
    XU_t = [nlSimContMOC([x, y, 0.0f0]) for x in xxFine, y in yyFine];
end
tSpan = 0:tInt:tEnd;
U0_sol = [XU_t[i,j][1][3] for i in 1:nEvalFine, j in 1:nEvalFine]; # zero everywhere
for (t, tVal) in enumerate(tSpan)
    local X1grid = [XU_t[i,j][t][1] for i in 1:nEvalFine, j in 1:nEvalFine];
    local X2grid = [XU_t[i,j][t][2] for i in 1:nEvalFine, j in 1:nEvalFine];
    if propRHO
        local RHOgrid = [(XU_t[i,j][t][3]) for i in 1:nEvalFine, j in 1:nEvalFine];
    else
        local RHOgrid = [RHO0[i,j]*exp(XU_t[i,j][t][3]) for i in 1:nEvalFine, j in 1:nEvalFine];
    end
    local normC = trapz((X1grid[:,1], X2grid[2,:]), RHOgrid)
    @show normC;
    # normC_str = @sprintf("|ρ| = %.3f",normC);
    figure(45,(8,4)); clf();
    subplot(1,2,1);
    pcolor(X1grid, X2grid, RHOgrid); colorbar();
    xlabel("x1"); ylabel("x2");
    xlim(minval, maxval);
    ylim(minval,  maxval);
    title(string("ρ(x1,x2) at t = $(tVal);"));#, normC_str));
    suptitle("Propagating ρ using MOC");
    tight_layout();

    ## Marginal PDF ρ(x1) = ∫_{x2Domain} ρ_{x1,x2} dx2 (not normalized)
    RHO_X1 = similar(xxFine); 
    tol = 5e-3; # 
    for (xInd,xVal) in enumerate(xxFine)
        ind = findall(z->abs(z-xVal) < tol, X1grid); # find all x1 in new grid close to xVal
        RHO_X1[xInd] = trapz((X2grid[ind]), (RHOgrid[ind])); # integrate numerically
    end
    subplot(1,2,2); 
    scatter(xxFine, RHO_X1, color = "r"); 
    xlabel("x1"); ylabel("ρ(x1)");
    tight_layout();

    # savefig("figs_moc/moc_mm/t$(t).png");
    # sleep(0.1);
end
##
if propRHO
    XUEND = [nlSimContMOC([x, y, rho0([x,y]) ])[end] for x in xxFine, y in yyFine];
else
    XUEND = [nlSimContMOC([x, y, 0.0f0 ])[end] for x in xxFine, y in yyFine];
end

X1END = [XUEND[i,j][1] for i in 1:nEvalFine, j in 1:nEvalFine];
X2END = [XUEND[i,j][2] for i in 1:nEvalFine, j in 1:nEvalFine];
if propRHO
    RHOEND = [(XUEND[i,j][3]) for i in 1:nEvalFine, j in 1:nEvalFine];
else
    RHOEND = [(RHO0[i,j]*exp(XUEND[i,j][3])) for i in 1:nEvalFine, j in 1:nEvalFine];
end


##
figure(87, (8,4)); clf();
# subplot(1,2,2);
surf(X1END, X2END, RHOEND);#, shading = "auto", cmap = "inferno");
# colorbar();
title("ρ at t = $(tEnd)");
xlabel("x1");
ylabel("x2");
xlim(minval, maxval);
ylim(minval,  maxval);
# title("MOC with RT dynamics");
tight_layout();

figure(243); clf();
contour(X1END, X2END, RHOEND);
title("ρ at t = $(tEnd)");
colorbar();
xlabel("x1");
ylabel("x2");
xlim(minval, maxval);
ylim(minval,  maxval);
tight_layout();

# XSOL = nlSimContMOC([-2,-2, 1/nGridPnts]);
# x1Sol = [XSOL[t][1] for t in 1:length(XSOL)];
# x2Sol = [XSOL[t][2] for t in 1:length(XSOL)];

# figure(78); clf();
# scatter(x1Sol[1], x2Sol[1], color = "r");
# scatter(x1Sol[2:end], x2Sol[2:end], color= "b");
# xlim(minval, maxval);
# ylim(minval,  maxval);


# figure(1189); clf();
# scatter(X1END, X2END);
# # scatter(X1END[1,1], X2END[1,1], color = "red");
# # scatter(X1END[2500], X2END[2500], color = "red");
# xlim(minval, maxval);
# ylim(minval,  maxval);
# end

## cheating, counting in bins
if simMOC_c
    nbin = 100;
    xp=collect(LinRange(minval,maxval,nbin+1));
    yp=collect(LinRange(minval,maxval,nbin+1));
    dx=xp[2] - xp[1];
    dy=dx;
    xcent = Vector{Float64}(undef, nbin); ycent = similar(xcent);
    xcent[1:nbin]=0.5*(xp[1:nbin]+xp[2:nbin+1]);
    ycent[1:nbin]=0.5*(yp[1:nbin]+yp[2:nbin+1]);
    PDF=zeros(nbin,nbin);
    for i=1:length(X1END)
        ii= Int64(min(max(ceil((X1END[i] -(minval))/dx),1),nbin)); # Because of meshgrid
        jj= Int64(min(max(ceil((X2END[i] -(minval))/dy),1),nbin)); 
        PDF[ii,jj] += RHOEND[i];
    end
    XCENT = zeros(nbin, nbin); YCENT = similar(XCENT);
    for i = 1:nbin, j = 1:nbin
        XCENT[i, j] = xcent[i]
        YCENT[i, j] = ycent[j]
    end
    normC = trapz((xcent, ycent), PDF); #sum(sum(PDF));
    ##
    PDFc = PDF/normC;
    figure(450); clf();
    pcolor(XCENT, YCENT, PDF,shading = "auto")
    # pcolor(XCENT, YCENT, PDF/normC,shading = "auto", cmap = "inferno")
    colorbar();
end
## simulate regular dynamics for a random initial state
if simDyn
    function nlSimMOC(x0)
        odeFn(x,p,t) = f(x)
        prob = ODEProblem(odeFn, x0, (0.0, tEnd));
        sol = solve(prob, Tsit5(), reltol = 1e-3, abstol = 1e-3);
        return sol
    end
    x0_tmp = -4.0f0 .+ 8.0f0*rand(2);
    sol_tmp = nlSimMOC(x0_tmp);

    # ##
    x1Sol = [sol_tmp.u[t][1] for t in 1:length(sol_tmp.t)];
    x2Sol = [sol_tmp.u[t][2] for t in 1:length(sol_tmp.t)];

    figure(98); clf();
    scatter(x1Sol[1], x2Sol[1], color = "r");
    scatter(x1Sol[2:end], x2Sol[2:end], s=5, color= "b");
    xlim(minval, maxval);
    ylim(minval,  maxval);
end

## save data
# using JLD2
# saveFile="dataTS_grid/moc.jld2";
# jldsave(saveFile;XU_t);


# #region
# ## MoC stuff
# using ForwardDiff
# # using ModelingToolkit
# # @parameters x1, x2, t
# # @variables ρ2(..)

# # x = [x1;x2]
# ρFn(x) = exp(first(phi[1](x, optParam))); 
# uFn(x) = first(phi[2](x, optParam)); 
# f(x) = [x[2]; -x[1] + (1-x[1]^2)*x[2] + uFn(x)];
# df(x) = ForwardDiff.jacobian(f,x);

# function rhoDyn(rho, x)
#     # dRho_dx = ForwardDiff.gradient(ρFn,x);
#     # df_dx = ForwardDiff.jacobian(f,x);
#     # dRho_dt = - ρFn(x)*tr(df_dx) - dot(f(x),dRho_dx);
#     dRho_dt = -rho*tr(df(x));
#     return dRho_dt;
# end

# function nlSimContMOC(x0)
#     # function to simulate nonlinear controlled dynamics with initial condition x0 and controller K
#     odeFn(x,p,t) = [f(x[1:2]); rhoDyn(x[end],x[1:2])];
#     prob = ODEProblem(odeFn, x0, (0.0, tEnd));
#     sol = solve(prob, Tsit5(), reltol = 1e-6, abstol = 1e-6);
#     return sol
# end
# nGridPnts = (length(xs2) + length(ys2));
# tcx0 = [tx; 1/nGridPnts];
# solSimContMOC = nlSimContMOC(tcx0);


# eq_rhs = sum([Differential(x[i])(F[i]) for i in 1:length(x)]);
# using ForwardDiff
# rhoNN_Fn = (x1,x2) -> exp(first(phi[1]([x1,x2], optParam)));
# tx1 = 0.0f0; tx2 = 1.0f0;
# drho_dx1 = ForwardDiff.derivative(z->rhoNN_Fn(z,tx2), tx1)
#endregion

## marginal pdf attempt 
# ts = 5; # time index
# tx2 = [XU_t[i,j][ts][1] for i in 1:nEvalFine, j in 1:nEvalFine];
# ux2 = [XU_t[i,j][ts][2] for i in 1:nEvalFine, j in 1:nEvalFine];
# px2 = [XU_t[i,j][ts][3] for i in 1:nEvalFine, j in 1:nEvalFine];

# figure(46);clf();
# pcolor(tx2, ux2, px2); colorbar();
# xlabel("x1"); ylabel("x2");
# tight_layout();

# tol = 5e-3;
# rx1_t2 = similar(xxFine);
# for (xInd,x) in enumerate(xxFine)
#     ind = findall(y->abs(y-x) < tol, tx2);
#     ind_p = sortperm(ux2[ind]);
#     rx1_t2[xInd] = trapz((ux2[ind])[ind_p], (px2[ind])[ind_p]);
# end
# figure(23);clf();
# scatter(xxFine, rx1_t2);

# minimum(rx1_t2)



