## test 2D integration using SciML-quadrature
minval = 0.0f0; maxval = 3.0f0; # for MM Model, use tEnd = 1.0
mu2 = [2,2]; # for mm model
sig2 = [1/8; 1/40] .* 1.0I(2); # for mm model
xxFine = range(minval, maxval, length = nEvalFine);
yyFine = range(minval, maxval, length = nEvalFine);
rh2d(x) = exp(-1 / 2 * (x - mu2)' * inv(sig2) * (x - mu2)) / (2 * pi * sqrt(det(sig2))); # ρ at t0
RH2 = [rh2d([x,y]) for x in xxFine, y in yyFine ];

qFn(x,p) = rh2d(x);
prob = QuadratureProblem(qFn, [minval, minval], [maxval, maxval]);
# sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3); # works
# sol = solve(prob, CubatureJLh(), reltol = 1e-3, abstol = 1e-3); # works
# sol = solve(prob, CubatureJLp(), reltol = 1e-3, abstol = 1e-3); # works
# sol = solve(prob, CubaVegas(), reltol = 1e-3, abstol = 1e-3); # doesn't work
# sol = solve(prob, CubaSUAVE(), reltol = 1e-3, abstol = 1e-3); # works
# sol = solve(prob, CubaDivonne(), reltol = 1e-3, abstol = 1e-3); #  works
# sol = solve(prob, CubaCuhre(), reltol = 1e-3, abstol = 1e-3); # works
# sol = solve(prob, VEGAS(), reltol = 1e-3, abstol = 1e-3); # works
@show sol.u