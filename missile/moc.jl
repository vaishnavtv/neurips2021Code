# solving missile fpke using MOC
# Pg. 25 from Andrea Weiss's thesis
using LinearAlgebra, DifferentialEquations, ForwardDiff
using PyPlot; pygui(true);
cd(@__DIR__);
include("cpu_missileDynamics.jl");

tEnd = 0.1;
# f(x) = [x[2]; (-x[1] + (1 - x[1]^2) * x[2])];
df(x) = ForwardDiff.jacobian(f,x);
rhoDyn(rho, x) = -rho*tr(df(x));
function nlSimContMOC(x0)
    odeFn(xu,p,t) = [f(xu[1:2]); rhoDyn(xu[end], xu[1:2])]
    prob = ODEProblem(odeFn, x0, (0.0, tEnd));
    sol = solve(prob, Tsit5(), reltol = 1e-3, abstol = 1e-3);
    return sol.u[end]
end
nEvalFine = 100; 
nGridPnts = nEvalFine^2;
# x0 = xTrim_Ma; u0 = 1/nGridPnts;
# txu0 = [x0;u0];
# solSimContMOC = nlSimContMOC(txu0);

minM = 1.2; maxM = 2.5;
minα = -1.0; maxα = 1.5;
xxFine = range(minM, maxM, length = nEvalFine);
yyFine = range(minα, maxα, length = nEvalFine);
XXFine = zeros(nEvalFine, nEvalFine); YYFine = similar(XXFine);
for i = 1:nEvalFine, j = 1:nEvalFine
    XXFine[i, j] = xxFine[i]
    YYFine[i, j] = yyFine[j]
end

XU0 = [[x, y, 1/nGridPnts] for x in xxFine, y in yyFine]; 
U0 = [1/nGridPnts for x in xxFine, y in yyFine];
XUEND = [nlSimContMOC([x, y, 1/nGridPnts]) for x in xxFine, y in yyFine];
UEND = [XUEND[i,j][3] for i in 1:nEvalFine, j in 1:nEvalFine];
##
figure(87, (8,4)); clf();
subplot(1,2,1);
pcolor(XXFine, YYFine, U0, shading = "auto", cmap = "inferno");
colorbar();
title("t=0")
xlabel("M");
ylabel("α");

subplot(1,2,2);
pcolor(XXFine, YYFine, UEND, shading = "auto", cmap = "inferno");
colorbar();
title("t = $(tEnd)");
xlabel("M");
ylabel("α");
# title("MOC with RT dynamics");
tight_layout();
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

