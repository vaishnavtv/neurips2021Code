using Flux, ProgressMeter, LinearAlgebra

# ========  Some parameterized basis functions =========

# Gaussian RBF
struct GaussianRBF β; xc; 
    GaussianRBF(d::Int;init=Flux.glorot_uniform) = new(init(1),init(d));
end
Flux.@functor GaussianRBF;
(f::GaussianRBF)(x::Vector) = exp.(-abs(f.β[1]) * sum((x .- f.xc) .* (x .- f.xc)));

# NNRBF
tanhScaled(x) = 0.5f0*(1.0f0+tanh(x));
struct NNRBF w1; xc1; w2; xc2;
    NNRBF(w1,xc1,w2,xc2) = new(w1, xc1, w2, xc2);
end
Flux.@functor NNRBF;
(f::NNRBF)(x) = prod(tanhScaled.(f.w1*(x .- f.xc1)) .* tanhScaled.(f.w2*(x .- f.xc2)));

# Linear Combination
struct Linear a;
    Linear(a) = new(a);
end
Flux.@functor Linear;
(f::Linear)(x) = f.a*x;

struct LinearPos a;
    LinearPos(a) = new(a);
end
Flux.@functor LinearPos;
(f::LinearPos)(x) = (f.a.^2)*x;

# Convex Combination
struct Convex a;
    Convex(d) = new(a);
end
Flux.@functor Convex;
(f::Convex)(x) = (f.a).^2*x/sum((f.a).^2);

function eye(d)
    return Diagonal(ones(d));
end

# Find optimal parameters.
function optimizeParameters(fhat, X, Ydata, costFcn, nIter,η; sequential=false)
    errorTraj = [];

    opt = ADAM(η, (0.9, 0.8));
    p = Progress(nIter);
    ps = Flux.params(fhat);

    for i in 1:nIter
        if sequential # Sequential Optimization
            for d in zip(X,Ydata)
                # grads = gradient(() -> Flux.mse(fhat(d[1]), d[2],agg=Flux.mean), ps);
                grads = gradient(() -> costFcn(fhat(d[1]), d[2]), ps);
                Flux.Optimise.update!(opt, ps, grads);
            end
        else # Batch Optimization
            grads = gradient(() -> costFcn(vcat(fhat.(X)...), Y), ps);
            Flux.Optimise.update!(opt, ps, grads);
        end
        ProgressMeter.next!(p,showvalues=[(:iter,i), (:loss,costFcn(vcat(fhat.(X)...),Y))]);
        push!(errorTraj,costFcn(vcat(fhat.(X)...), vcat(Ydata...)));
    end
    optimalCost = costFcn(vcat(fhat.(X)...), vcat(Ydata...));
    return optimalCost, errorTraj;
end



