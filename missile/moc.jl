# solving missile fpke using MOC
# Pg. 25 from Andrea Weiss's thesis
using LinearAlgebra, DifferentialEquations, ForwardDiff, Trapz
using PyPlot; pygui(true);
cd(@__DIR__);
include("cpu_missileDynamics.jl");
simMOC = true;
simDyn = false;
tEnd = 2.0; tInt = 0.05; #tEnd/100;

df(x) = ForwardDiff.jacobian(f,x);
uDyn(rho, x) = -tr(df(x)); 
# rhoDyn(rho, x) = -rho*tr(df(x));
minM = 1.0; maxM = 2.5;
minα = -1.0; maxα = 1.5;

# propagate density using MOC
# if simMOC 
function nlSimContMOC(x0)
    odeFn(xu,p,t) = [f(xu[1:2]); uDyn(xu[end], xu[1:2])]
    # odeFn(xu,p,t) = [f(xu[1:2]); rhoDyn(xu[end], xu[1:2])]
    prob = ODEProblem(odeFn, x0, (0.0, tEnd));
    sol = solve(prob, Tsit5(), saveat = tInt, reltol = 1e-3, abstol = 1e-3);
    return sol.u
end
##
nEvalFine = 100; # initial grid, no. of points along each axis
nGridPnts = nEvalFine^2;

xxFine = range(minM, maxM, length = nEvalFine);
yyFine = range(minα, maxα, length = nEvalFine);
XXFine = zeros(nEvalFine, nEvalFine); YYFine = similar(XXFine);
for i = 1:nEvalFine, j = 1:nEvalFine
    XXFine[i, j] = xxFine[i]
    YYFine[i, j] = yyFine[j]
end

μ_ss = xTrim_Ma
Σ_ss = 1.0*[1; 1] .* 1.0I(2);

rho0(x) =  exp(-1 / 2 * (x - μ_ss)' * inv(Σ_ss) * (x - μ_ss)) / (2 * pi * sqrt(det(Σ_ss))); # ρ at t0
# rho0(x) = 1/((maxM - minM)*(maxα - minα));
# rho0(x) = 0.0f0;
RHO0 = [rho0([x,y]) for x in xxFine, y in yyFine ];
norm_RHO0 = trapz((xxFine, yyFine), RHO0);
@show norm_RHO0
XU0 =  [[x, y, 0.0f0] for x in xxFine, y in yyFine];
# U0 = [1/nGridPnts for x in xxFine, y in yyFine];
##
XU_t = [nlSimContMOC([x, y, 0.0f0]) for x in xxFine, y in yyFine];

tSpan = 0:tInt:tEnd;
for (t, tVal) in enumerate(tSpan)
    local X1grid = [XU_t[i,j][t][1] for i in 1:nEvalFine, j in 1:nEvalFine];
    local X2grid = [XU_t[i,j][t][2] for i in 1:nEvalFine, j in 1:nEvalFine];
    local Ugrid = [RHO0[i,j]*exp(XU_t[i,j][t][3]) for i in 1:nEvalFine, j in 1:nEvalFine];
    local normC = trapz((X1grid[:,1], X2grid[2,:]), Ugrid)
    # @show normC;
    figure(45); clf();
    contourf(X1grid, X2grid, Ugrid); colorbar();
    xlabel("x1"); ylabel("x2");
    # xlim(minM, maxM);
    # ylim(minα, maxα);
    title("Propagating ρ using MOC; t = $(tVal)");
    sleep(0.1);
end
##


X1END = [XU_t[i,j][end][1] for i in 1:nEvalFine, j in 1:nEvalFine];
X2END = [XU_t[i,j][end][2] for i in 1:nEvalFine, j in 1:nEvalFine];

UEND = [XU_t[i,j][end][3] for i in 1:nEvalFine, j in 1:nEvalFine];
RHOEND = [RHO0[i,j]*exp(UEND[i,j])  for i in 1:nEvalFine, j in 1:nEvalFine];

##
# figure(87, (8,4)); clf();
# subplot(1,2,2);
# surf(X1END, X2END, RHOEND);#, shading = "auto", cmap = "inferno");
# # colorbar();
# title("ρ at t = $(tEnd)");
# xlabel("x1");
# ylabel("x2");
# xlim(minM, maxM);
# ylim(minα, maxα);
# # title("MOC with RT dynamics");
# tight_layout();

figure(243); clf();
contourf(X1END, X2END, RHOEND);
title("ρ at t = $(tEnd)");
colorbar();
xlabel("x1");
ylabel("x2");
# xlim(minM, maxM);
# ylim(minα, maxα);
tight_layout();

## simulate regular dynamics for a random initial state
simDyn = true;
if simDyn
    function nlSimMOC(x0)
        odeFn(x,p,t) = f(x)
        prob = ODEProblem(odeFn, x0, (0.0, tEnd));
        sol = solve(prob, Tsit5(), saveat = tInt, reltol = 1e-3, abstol = 1e-3);
        return sol
    end
    x0_tmp = [minM; minα] .+ [maxM - minM; maxα - minα].*rand(2);
    sol_tmp = nlSimMOC(x0_tmp);

    # ##
    x1Sol = [sol_tmp.u[t][1] for t in 1:length(sol_tmp.t)];
    x2Sol = [sol_tmp.u[t][2] for t in 1:length(sol_tmp.t)];

    figure(98); clf();
    scatter(x1Sol[1], x2Sol[1], color = "r");
    scatter(x1Sol[2:end], x2Sol[2:end], s = 5, color= "b");
    xlim(minM, maxM);
    ylim(minα,  maxα);
end

