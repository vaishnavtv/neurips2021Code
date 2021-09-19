using LinearAlgebra
using CUDA, Flux
CUDA.allowscalar(false)
xTrim = Float32.([1.299897814066209,   0.072051923467205,  -0.414704022535722,   0.063896856742446]) #|> gpu;
uTrim = -0.222907270172726f0 #|> gpu;
K = -Float32.([-0.302818617638008,   0.044838927180658]) #|> gpu;
ii = [1,2];
xTrim_Ma = Float32.([1.299897814066209,   0.072051923467205]) #|> gpu;
xTrim_g = cu(xTrim);
K_g = cu(K); xTrim_Ma_g = cu(xTrim_Ma);
##
mask1 = [1 0] |> gpu;
mask2 = [0 1] |> gpu;
mask3 = [0 0 1 0] |> gpu;
mask4 = [0 0 0 1] |> gpu;
function dynamicsMissile(x, u)
    # mask1 = [1 0] |> gpu;
    # mask2 = [0 1] |> gpu;
    # mask3 = [0 0 1 0] |> gpu;
    # mask4 = [0 0 0 1] |> gpu;
    if isa(x, CuArray)
        M = sum(mask1*x)#[1];
        a = sum(mask2*x)#[2];
        g = sum(mask3*xTrim_g)#[3];
        Q = sum(mask4*xTrim_g)#[4];
    else
        M = x[1];
        a = x[2];
        g = xTrim[3];
        Q = xTrim[4];
    end
    abs_a = a*sign(a) #sqrt(a^2)

    Mdot = 0.4008f0*M^2*a^3*sin(a)-0.6419f0*M^2*abs_a*a*sin(a)-0.2010f0*M^2*(2f0-M/3f0)*a*sin(a)-0.0062f0*M^2-0.0403f0*M^2*sin(a)*u-0.0311f0*sin(g);
    αdot = 0.4008f0*M*a^3*cos(a)-0.6419f0*M*abs_a*a*cos(a)-0.2010f0*M*(2f0-M/3f0)*a*cos(a)-0.0403f0*M*cos(a)+0.0311f0*cos(g)/M + Q;

    xdot = CUDA.adapt(DiffEqBase.parameterless_type(x),[Mdot,αdot]) 
    return xdot;

end
##
function f(x) # controlled nonlinear dynamics
    if isa(x, CuArray)
        uDel = dot((K_g),(x- (xTrim_Ma_g)))
    else
        uDel = dot((K),(x- (xTrim_Ma)))
    end
    u = uTrim + uDel  #(K*(x - xTrim_Ma)); #xTrim[ii]));
    # @show u
    return -dynamicsMissile(x,u); # returns M, α (reverse time)
end

function g(x)
    out = Float32.([0.0; 0.0].*1.0I(2)); # diffusion in α
    return CUDA.adapt(DiffEqBase.parameterless_type(x),out) 
    # return [0.0; 1.0].*1.0I(2); # diffusion in alpha
    # M = x[1]; a = x[2];
    # return [-0.0403f0*M^2*sin(a); 0.0f0] 
end