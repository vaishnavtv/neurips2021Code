using MosekTools, PyPlot, GLPK, Clp
include("rbfNet.jl"); # Load utility files.

# pygui(:qt);

N = 2000;
maxval = 1;
xmin = maxval * [-1, -1];
xmax = maxval * [1, 1];
alg = :halton;
Ci = HyperCubeSampler(xmin, xmax, N, algo=alg); 

w1 = ones(N); w1 = w1 / sum(w1); # Uniform distribution.

w2 = similar(w1);
xc = [-1 -1 1 1 ;
      -1  1 -1 1]*0.25;

xc = [-1 1 ;
      -1 1]*0.25;      

min = 0.5;
max = 0.65;

for i in 1:N
    C = [(norm(Ci[:,i]-xc[:,j]) <= max && norm(Ci[:,i]-xc[:,j]) >= min) for j in 1:size(xc,2)];
    if sum(map(|,C)) > 0
        w2[i] = 1;
    else
        w2[i] = 0;
    end
end
w2 = w2 / sum(w2); # Error distribution.

Optimizer = Mosek.Optimizer(LOG = 0); # Fast
# Optimizer = GLPK.Optimizer(); # Slow

# Generate samples using sinkhorn
@time Phi = otMap(Ci, w1, w2, Optimizer, alg=:sinkhorn,α=0.005);

## trying GPU
using CUDA
CUDA.allowscalar(false)
Phi_gpu = otMap_gpu(Ci, w1, w2, Optimizer, alg=:sinkhorn, maxIter = 10000, α=0.01f0);

# # Generate samples using LP (Earth Mover's Distance --> emd)
# @time Phi = otMap(Ci, w1, w2, Optimizer, alg=:emd);

y = Ci * (N * Phi'); # Need transpose on Phi, if using OptimalTransport.emd()

##
# using JLD2
# jldsave("data/sinkhorn.jld2"; Ci, y);


##
# figure(); clf();
# scatter(Ci[1,:],Ci[2,:],c="r",s=1);
# scatter(y[1,:],y[2,:],c="b",s=1);
# axis("square");

# Write a sampler paper.
# We need the sampling technique in many applications.
# Start tomorrow. Look at scenario optimization book for summary of the sampling techniques.
