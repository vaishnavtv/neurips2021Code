using LinearAlgebra
xTrim = Float32.([1.299897814066209   0.072051923467205  -0.414704022535722   0.063896856742446]);
uTrim = -0.222907270172726f0
K = -Float32.([-0.302818617638008   0.044838927180658]);
ii = [1,2];
xTrim_Ma = Float32.([1.299897814066209,   0.072051923467205]);
##

function dynamicsMissile(x, u)
    M = x[1];
    a = x[2];
    g = xTrim[3];
    Q = xTrim[4];
    abs_a = a*sign(a) #sqrt(a^2)
    xdot = Vector(undef, length(x));

    Mdot = 0.4008f0*M^2*a^3*sin(a)-0.6419f0*M^2*abs_a*a*sin(a)-0.2010f0*M^2*(2f0-M/3f0)*a*sin(a)-0.0062f0*M^2-0.0403f0*M^2*sin(a)*u-0.0311f0*sin(g);
    αdot = 0.4008f0*M*a^3*cos(a)-0.6419f0*M*abs_a*a*cos(a)-0.2010f0*M*(2-M/3)*a*cos(a)-0.0403f0*M*cos(a)+0.0311f0*cos(g)/M + Q;

    # @show typeof(M)
    # @show typeof(u)
    # @show typeof(M*u)
    xdot = [Mdot,αdot]#[Mdot, αdot]
    return xdot;

end

##
function f(x) # controlled nonlinear dynamics
    u = uTrim + dot(K,(x- xTrim_Ma)) #(K*(x - xTrim_Ma)); #xTrim[ii]));
    # @show u
    return -dynamicsMissile(x,u); # returns M, α (reverse time)
end

function g(x)
    # return [0.0; 1.0].*1.0I(2); # diffusion in alpha
    M = x[1]; a = x[2];
    return [-0.0403f0*M^2*sin(a); 0.0f0] 
end