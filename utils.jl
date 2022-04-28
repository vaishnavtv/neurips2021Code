# Get initial set of params
using Distributions, LinearAlgebra, Flux#, CUDA
import Random:seed!; seed!(1);

function sub2ind(i, N)
    d = length(N)
    if d == 1
        return i
    else
        x = i - 1
        ii = zeros(Integer, d)
        for j = d:-1:1
            A = prod(N[1:(j - 1)])
            ii[j] = div(x, A) + 1
            x = x % A
        end
        return ii
    end
end

function GenerateNDGrid(lb, ub, N)
    d = length(N)

    if length(N) != length(lb) ||
       length(N) != length(ub) ||
       length(lb) != length(ub)
        error("Dimension mismatch.")
    end

    g = [range(lb[i], ub[i], length=N[i]) for i = 1:d]
    nmax = prod(N)
    G = zeros(d, nmax)

    for i = 1:nmax
        ii = sub2ind(i, N)
        for j = 1:d
            G[j, i] = g[j][ii[j]]
        end
    end
    return G
end

# Define domain and collocation points
# maxval = 4.0;
# xmin = maxval * [-1, -1];
# xmax = maxval * [1, 1];
# nGrid =100;
# C = GenerateNDGrid(xmin, xmax, nGrid*[1,1]);

# # # # Initial condition
# μ0  = [0,0]; Σ0 = 0.1*1.0I(2); #gaussian 
# r0 = pdf(MvNormal(μ0, Σ0),C);
# rho0 = reshape(r0, (1, nGrid^2));

# dim = 2 # number of dimensions
# nn = 100; activFunc = tanh;
# chain = Chain(Dense(dim,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,nn,activFunc), Dense(nn,1)) #|> gpu;

# loss(x,y) = Flux.Losses.mse(exp.(chain(x)),y);
# ps = Flux.params(chain);
# opt = ADAM(1e-3);
# train_loader = Flux.DataLoader(((C),(rho0)))#, batchsize=1, shuffle=true);
# maxIters = 5000;

# using BenchmarkTools
# @btime Flux.Optimise.train!(loss, ps, train_loader, opt) ;
# 8.669 s (24310716 allocations: 1.63 GiB) for batchsize = 1 on gpu
# 2.036 s (5930000 allocations: 3.51 GiB)  for batchsize = 1 on cpu
# 9.418 s (24323430 allocations: 1.63 GiB) for full batch on gpu
# 1.948 s (5930000 allocations: 3.51 GiB) for full batch on cpu 

# Flux.@epochs maxIters Flux.Optimise.train!(loss, ps, train_loader, ADAM(1e-3));


##
# XX = reshape(C[1, :], nGrid, nGrid);
# YY = reshape(C[2, :], nGrid, nGrid);
# using PyPlot; pygui(true);
# using Plots;
# p1 = heatmap(C[1,1:nGrid], C[1,1:nGrid], reshape(r0, (nGrid, nGrid)));
# # # p2 = heatmap(C[1,1:nGrid], C[1,1:nGrid], reshape(Array(exp.(chain((C)))), (nGrid, nGrid)));
# p2 = heatmap(C[1,1:nGrid], C[1,1:nGrid], reshape(Array((phi((C),th0))), (nGrid, nGrid)));
# p3 = heatmap(C[1,1:nGrid], C[1,1:nGrid], abs.(reshape(r0, (nGrid, nGrid)) - reshape(Array(phi((C),th0)), (nGrid, nGrid))));
# # p = plot(p1, p2, p3, aspect_ratio=:equal);
# p = plot(p2)
# display(p);
##
# figure(56, (12,4)); clf();
# subplot(1,3,1);
# pcolor(XX, YY, reshape(r0, (nGrid, nGrid)), shading = "auto", cmap = "inferno"); colorbar();
# tight_layout();
# subplot(1,3,2);
# pcolor(XX, YY, reshape(chain(C), (nGrid, nGrid)), shading = "auto", cmap = "inferno"); colorbar();
# tight_layout();
# subplot(1,3,3);
# pcolor(XX, YY, reshape(rho0 - chain(C), (nGrid, nGrid)), shading = "auto", cmap = "inferno"); colorbar();
# tight_layout();
