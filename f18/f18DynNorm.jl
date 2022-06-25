cd(@__DIR__);
include("f18Dyn.jl")
using LinearAlgebra

# Normalized state variables
vMin = 200f0; vMax = 1500f0;
αMin = -deg2rad(10f0); αMax = deg2rad(90f0);
θMin = -deg2rad(10f0); θMax = deg2rad(90f0);
qMin = Float32(-pi/3); qMax = Float32(pi/3);
##
# How to normalize
vN(v) = (v - vMin)/(vMax - vMin); # v between 200 and 1500
alpN(α) = (α - (αMin))/(αMax - αMin) # α between -10 and 90
thN(θ) = (θ - (θMin))/(θMax - θMin) # θ between -10 and 90
qN(q) = (q - (qMin))/(qMax - qMin) # -60 to 60 deg/s

An = 1.0f0I(4);
bn = zeros(Float32,4);

An[1,1] = 1/(vMax - vMin); bn[1] = -vMin/(vMax - vMin);
An[2,2] = 1/(αMax - αMin); bn[2] = -αMin/(αMax - αMin);
An[3,3] = 1/(θMax - θMin); bn[3] = -θMin/(θMax - θMin) ;
An[4,4] = 1/(qMax - qMin); bn[4] = - qMin/(qMax - qMin) ;

AnInv = inv(An);
# xnTrim = An*f18_xTrim[indX] + bn
# xOrig = An\(xn - bn)

# ====== HOW YOU WANT NORMALIZED DYNAMICS TO LOOK
# maskTrim = ones(Float32,length(f18_xTrim)); maskTrim[indX] .= 0f0;
# function f18n(xn)
#     # normalized input to f18 dynamics (full dynamics)
#     xi = An\(xn - bn); # x of 'i'nterest
#     ui = [Kc1(xi...), Kc2(xi...)];

#     xFull = maskTrim.*f18_xTrim + maskIndx*xi;
#     uFull = [1f0;1f0;0f0;0f0].*f18_uTrim + maskIndu*ui; 

#     xdotFull = f18Dyn(xFull, uFull)
#     return An*(xdotFull[indX]) # return the 4 state dynamics
# end

# ==== HOW TO COMPUTE SIGMA =====
sig_v = (vN(f18_xTrim[indX][1] + 20f0) - vN(f18_xTrim[indX][1] - 20f0))/6f0
sig_alp = (alpN(f18_xTrim[indX][2] + deg2rad(2f0)) - alpN(f18_xTrim[indX][2] - deg2rad(2f0)))/6f0
sig_th = (thN(f18_xTrim[indX][3] + deg2rad(2f0)) - thN(f18_xTrim[indX][3] - deg2rad(2f0)))/6f0
sig_q = (qN(f18_xTrim[indX][4] + deg2rad(2f0)) - qN(f18_xTrim[indX][4] - deg2rad(2f0)))/6f0

