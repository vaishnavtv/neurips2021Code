using QuasiMonteCarlo,
    Convex, MosekTools, NearestNeighbors, OptimalTransport, Primes, GoldenSequences

function otMap(X, w1, w2, Optimizer; alg = :emd, maxIter = 1000, α = 0.01) # X is dxN matrix, d in dimension, N is number of samples. W1, W2 are column vectors
    # Use sparse variables -- that might speed up things.
    n = size(X, 2)

    C = zeros(n, n)
    for i = 1:n
        for j = 1:n
            dx = X[:, i] - X[:, j]
            C[i, j] = norm(dx)^2
        end
    end

    # Call the code from OptimalTransport.jl
    if alg == :emd
        Phi = emd(w1, w2, C, Optimizer)
    elseif alg == :sinkhorn
        Phi = sinkhorn(w1, w2, C, α, maxiter = maxIter)
    elseif alg == :sinkhorn_stab
        Phi = sinkhorn_stabilized(w1, w2, C, α, max_iter = maxIter)
    else
        error("Unsupported algorithm specified")
    end
    return Phi
end

# Convert linear index to multi-dimensional index
# ===============================================
function sub2ind(i, N)
    d = length(N)
    if d == 1
        return i
    else
        x = i - 1
        ii = zeros(Integer, d)
        for j = d:-1:1
            A = prod(N[1:(j-1)])
            ii[j] = div(x, A) + 1
            x = x % A
        end
        return ii
    end
end

# Generate ND grid
# ================
"""
G = GenerateNDGrid(lb,ub,N)
lb, ub, N are d-dimensional vectors specifying the limits and number of points along each dimension.
G is d x prod(N) matrix with grid points.
"""
function GenerateNDGrid(lb, ub, N)
    d = length(N)

    if length(N) != length(lb) || length(N) != length(ub) || length(lb) != length(ub)
        error("Dimension mismatch.")
    end

    g = [range(lb[i], ub[i], length = N[i]) for i = 1:d]
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

# Only 2D ... obsolete ... use GenerateNDGrid above.
function collocationGrid(x, y, N)
    X = range(x[1], x[2], length = N)
    Y = range(y[1], y[2], length = N)
    Cs = []
    for i = 1:N, j = 1:N
        push!(Cs, [X[i], Y[j]])
    end
    return hcat(Cs...)
end

# Gaussian RBF
function GaussRBF(x, xc, β)
    return exp(-β * sum((x .- xc) .^ 2))
end

# Get the corner points of the hypercube defined by x1,x2
function cornerPoints(xmin, xmax)
    x1 = xmin[:]
    x2 = xmax[:] # Vectorize
    d = length(x1)
    if d != length(x2)
        error("dimension mismatch in x1, x2")
    end

    X = [x1 x2] # d x 2 matrix
    p = [] # Not the best way to do it.

    ii = zeros(Int, d)
    for i = 0:(2^d-1)
        digits!(ii, i, base = 2)
        xx = zeros(Float64, d)
        for j = 1:d
            xx[j] = X[j, ii[j]+1]
        end
        push!(p, xx)
    end
    return hcat(p...)
end

# Generate samples in a hypercube.
"""
X = HyperCubeSampler(xmin,xmax,N; algo=:sobol)
xmin and xmax d-dimensional vectors specifying the limits along each dimension.
N is the total number of points generated in the domain
algo ∈ [:sobol,:halton, :golden], for Sobol sequence, Halton sequence, or Golden sequence
X is d x N matrix with grid points.
"""
function HyperCubeSampler(xmin, xmax, N; algo = :sobol)
    if (length(xmin) != length(xmax))
        error("xmin and xmax must be vectors of equal length")
    end

    d = length(xmin)
    algoSet = [:sobol, :halton, :golden]

    if algo ∉ algoSet
        error("algo must be in $algoSet")
    end

    if algo == :sobol
        # Generate Sobol sequence
        X = QuasiMonteCarlo.sample(N, xmin, xmax, SobolSample())
    elseif algo == :halton
        # Generate Halton sequence
        primeNums = [prime(i + 1) for i = 1:d]
        X = QuasiMonteCarlo.sample(N, xmin, xmax, LowDiscrepancySample(primeNums))
    elseif algo == :golden
        S = collect(Iterators.take(GoldenSequence(d), N))
        X = xmin .+ Diagonal(xmax - xmin) * hcat([collect(x) for x in S]...)
    end
    return X
end

# Determine shape function width from collocation points
function findβ(C, n)
    kdtree = KDTree(C)
    idxs, dists = knn(kdtree, C, n)
    dmax = maximum.(dists)
    β = 2 ./ dmax # It is possible that dmax = 0. We need to check this. Although rarely ...
    return β
end


#= 
Determine indices of boundary and interiot points from scattered data points
Boundary is determined from a points that define a hypercube. =#
function BoundaryAndInteriorNodes(X, domainBoundary)
    bddI = []
    d, N = size(X)
    for i = 1:N
        for j = 1:d
            if (X[j, i] == domainBoundary[j, 1] || X[j, i] == domainBoundary[j, 2])
                push!(bddI, i)
            end
        end
    end

    boundaryNodes = Integer.(unique(bddI))
    interiorNodes = setdiff(1:N, boundaryNodes)

    return boundaryNodes, interiorNodes
end


# Evaluate the approximated function.
function ρhat(α, Xc, β, x)
    nRBF = size(Xc, 2)
    return sum([α[i] * GaussRBF(x, Xc[:, i], β[i]) for i = 1:nRBF])
end


# Solve the PDE as an optimization 
function solvePDE(pdeError, rbfX, collocX, domain, Optimizer)

    d = size(rbfX, 1)
    β = findβ(rbfX, 2^d) # Uses nearest neighbour

    nRBF = size(rbfX, 2)
    nColloc = size(collocX, 2)

    A = zeros(nColloc, nRBF)
    for i = 1:nColloc
        for j = 1:nRBF
            val = pdeError(collocX[:, i], rbfX[:, j], β[j])
            A[i, j] = val
        end
    end

    ib, ii = BoundaryAndInteriorNodes(collocX, domain)

    # Evaluate the collocation matrix at the boundary points.
    B = zeros(length(ib), nRBF)
    for i = 1:length(ib)
        for j = 1:nRBF
            B[i, j] = GaussRBF(collocX[:, ib[i]], rbfX[:, j], β[j])
        end
    end

    ## ======================== Solve optimization problem ========================
    # -- Need to directly call solver instead of going through Convex. Will there be any performance benefit?
    # JuMP was too slow ... for Matrix/Vector computations.
    # Convex is fast, but gets slower as problem size grows.
    # Call CG, or write my own solver?
    # This needs to be figure out ... otherwise won't scale to high dimension.
    # =============================================================================
    α = Convex.Variable(nRBF, Positive())
    e = A * α

    problem =
        Convex.minimize(norm(e, 2) + 0.0001 * norm(α, 2), [B * α == 0.0, sum(α) == 1.0])

    @time Convex.solve!(problem, Optimizer)
    return α.value, norm(A * α.value, Inf), problem.status
end

function ErrorBasedSampler(
    nPoints,
    nSamples,
    rbfX,
    α,
    pdeError,
    domain,
    Optimizer;
    sampler = :halton,
    otAlgo = :LP,
)
    xmin = domain[:, 1]
    xmax = domain[:, 2]

    errorC = HyperCubeSampler(xmin, xmax, nSamples, algo = sampler)
    d = size(rbfX, 1)
    β = findβ(rbfX, 2^d)
    T =
        [
            pdeError(errorC[:, j], rbfX[:, i], β[i]) for j = 1:size(errorC, 2),
            i = 1:size(rbfX, 2)
        ] * α
    EqnError = T .^ 2

    sortedIndices = sortperm(EqnError[:])
    topErrorIndices = sortedIndices[end:-1:(end-nPoints+1)]
    topErrors = EqnError[topErrorIndices]
    topErrorLocations = errorC[:, topErrorIndices]

    w2 = topErrors
    w2 = w2 / sum(w2) # Error distribution.

    w1 = ones(nPoints)
    w1 = w1 / sum(w1) # Uniform distribution.

    Phi = otMap(topErrorLocations, w1, w2, Optimizer, alg = otAlgo)
    y = topErrorLocations * (length(w1) * Phi') # Need transpose on Phi, if using OptimalTransport.emd()
    return y
end


