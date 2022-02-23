using Symbolics, Flux

# Generate PDE -- Most General Setting -- Works for any dynamics.
function ForwardFPKE_Ito(d::Int;exponential=false)
    @variables x[1:d]
    x = Symbolics.scalarize(x)

    @variables η(..)
    η = η(x...)

    if exponential
        ρ = exp(-η); 
    else
        ρ = η
    end

    @variables F[1:d](x...)
    @variables G[1:d, 1:d](x...)
    F = Symbolics.scalarize(F)
    G = Symbolics.scalarize(G)

    FF = F * ρ
    GG = G * ρ

    # FPKE Equation
    T1 = sum([Differential(x[i])(FF[i]) for i in 1:length(x)])
    T2 = sum([(Differential(x[i]) * Differential(x[j]))(GG[i, j]) for j in 1:length(x) for i = 1:length(x)])
    if exponential
        Eqn = simplify(expand_derivatives(-T1 + T2) / ρ)
    else
        Eqn = simplify(expand_derivatives(-T1 + T2))
    end

    # Make list of all operators
    v0 = [η; F[:]; G[:]]
    grad_ρ = Symbolics.gradient(η, x)
    hess_ρ = Symbolics.jacobian(Symbolics.gradient(η, x), x) # Can't use hessian ... problem with Substitution due to symmetric hessian

    JacF = Symbolics.jacobian(F, x)
    JacVecG = Symbolics.jacobian(G[:], x)
    D2G = [(Differential(x[i]) * Differential(x[j]))(G[i, j]) for j in 1:length(x) for i = 1:length(x)]

    # Substitution variables
    @variables ρval, ∇ρ[1:d], ∇2ρ[1:d, 1:d], f[1:d], g[1:d, 1:d], jacf[1:d, 1:d], jacvecg[1:d^2, 1:d], d2g[1:d, 1:d]

    ∇ρ = Symbolics.scalarize(∇ρ)
    ∇2ρ = Symbolics.scalarize(∇2ρ)
    f = Symbolics.scalarize(f)
    g = Symbolics.scalarize(g)
    jacf = Symbolics.scalarize(jacf)
    jacvecg = Symbolics.scalarize(jacvecg)
    d2g = Symbolics.scalarize(d2g)

    # Now substitute
    V = [D2G; JacVecG[:]; JacF[:]; hess_ρ[:]; grad_ρ[:]; F[:]; G[:]; ρ]
    Vsub = [d2g[:]; jacvecg[:]; jacf[:]; ∇2ρ[:]; ∇ρ[:]; f[:]; g[:]; ρval]

    Eqn_sub = substitute(Eqn, V .=> Vsub)

    # Generate Function -- Has some @inbounds stuff.
    pde_error = build_function(Eqn_sub, ρval, ∇ρ, ∇2ρ, f, g, jacf, jacvecg, d2g, expression = Val{false}, target = Symbolics.JuliaTarget())
    return pde_error, Eqn_sub
end

# Symbolic Derivatives of NN functions
function NNFunc_and_Derivatives(F, x_sym; hessian = false) # derivatives w.r.t x
    p_num, fhat = Flux.destructure(F)
    uhat(p, x) = fhat(p)(x)[1];

    nParams = length(p_num)
    @variables p_sym[1:nParams]
    p_sym = Symbolics.scalarize(p_sym)

    uhat_sym = uhat(p_sym, x_sym)

    # if length(F.layers[end].b) > 1
    #     vecFunc = true
    #     if hessian
    #         println("Vector function detected ... Hessian option is ignored")
    #     end
    # else
    #     vecFunc = false
    # end

    ∇uhat_sym = Symbolics.gradient(uhat_sym, x_sym)
    ∇uhat_func = build_function(∇uhat_sym, p_sym, x_sym, expression = Val{false}, target = Symbolics.JuliaTarget())[1]

    ∇2uhat_sym = Symbolics.hessian(uhat_sym[1], x_sym)
    ∇2uhat_func = build_function(∇2uhat_sym, p_sym, x_sym, expression = Val{false}, target = Symbolics.JuliaTarget())[1]
    
    return (uhat, ∇uhat_func, ∇2uhat_func, vcat(p_num...))
    
end

function SymbolicDerivativesOfDynamics(f,g,Q,d)
    # Assumes f(x) and g(x) is a x-dependent only.
    @variables x[1:d]
    x = Symbolics.scalarize(x);
    F = f(x);
    G = g(x)*Q*g(x)';

    jacF = Symbolics.jacobian(F,x);
    jacVecG = Symbolics.jacobian(G[:],x);
    d2G = [expand_derivatives((Differential(x[i]) * Differential(x[j]))(G[i, j])) for j in 1:length(x), i = 1:length(x)];

    jacF_func = build_function(jacF, x, expression = Val{false}, target = Symbolics.JuliaTarget())[1]
    jacVecG_func = build_function(jacVecG, x, expression = Val{false}, target = Symbolics.JuliaTarget())[1]
    d2G_func = build_function(d2G, x, expression = Val{false}, target = Symbolics.JuliaTarget())[1]
    G_func = build_function(G, x, expression = Val{false}, target = Symbolics.JuliaTarget())[1]
    F_func = build_function(F, x, expression = Val{false}, target = Symbolics.JuliaTarget())[1]
    return F_func, G_func, jacF_func, jacVecG_func, d2G_func;
end

function AutoDerivativesOfDynamics(f,g,Q,d)
    # For dynamics that cannot be converted to symbolic functions ... e.g. dynamics with look up tables, etc.
    # Need to code this up.
end




