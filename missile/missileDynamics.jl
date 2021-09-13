using LinearAlgebra
xTrim = [1.299897814066209   0.072051923467205  -0.414704022535722   0.063896856742446];
uTrim = -0.222907270172726
K = -[-0.302818617638008   0.044838927180658];
ii = [1,2];
##

function dynamicsMissile(x, u)
    M = x[1];
    a = x[2];
    g = xTrim[3];
    Q = xTrim[4];
    abs_a = a*sign(a) #sqrt(a^2)
    xdot = Vector(undef, length(x));

    Mdot = 0.4008*M^2*a^3*sin(a)-0.6419*M^2*abs_a*a*sin(a)-0.2010*M^2*(2-M/3)*a*sin(a)-0.0062*M^2-0.0403*M^2*sin(a)*u[1]-0.0311*sin(g);
    αdot = 0.4008*M*a^3*cos(a)-0.6419*M*abs_a*a*cos(a)-0.2010*M*(2-M/3)*a*cos(a)-0.0403*M*cos(a)+0.0311*cos(g)/M + Q;

    xdot = [Mdot, αdot]
    return xdot;

end

##
function f(x) # controlled nonlinear dynamics
    u = uTrim .+ K*(x .- xTrim[ii]);
    return -dynamicsMissile(x,u); # returns M, α (reverse time)
end

function g(x)
    return [1.0; 1.0].*1.0I(2); # diffusion in both states
end