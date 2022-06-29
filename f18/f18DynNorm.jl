cd(@__DIR__);
include("f18Dyn.jl")
using LinearAlgebra

# Normalized state variables
vMin = 200f0; vMax = 1500f0;
αMin = -deg2rad(10f0); αMax = deg2rad(90f0);
θMin = -deg2rad(10f0); θMax = deg2rad(90f0);
qMin = Float32(-pi/3); qMax = Float32(pi/3);

## ============
# How to normalize to between [0,1]
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

## ========== 
# How to normalize all variables to [-bd,bd] where bd: bound
bd = 5.0f0;
vN2(v) = 2*bd*(v - (vMax+vMin)/2f0)/(vMax - vMin)
alpN2(α) = 2*bd*(α - (αMax + αMin)/2f0)/(αMax - αMin)
thN2(θ) = 2*bd*(θ - (θMax + θMin)/2f0)/(θMax - θMin)
qN2(q) = 2*bd*(q - (qMax + qMin)/2f0)/(qMax - qMin)

An2 = 1.0f0I(4);
bn2 = zeros(Float32,4);

An2[1,1] = 2*bd/(vMax - vMin); bn2[1] = -2*bd*((vMax+vMin)/2f0)/(vMax - vMin);
An2[2,2] = 2*bd/(αMax - αMin); bn2[2] = -2*bd*((αMax + αMin)/2f0)/(αMax - αMin);
An2[3,3] = 2*bd/(θMax - θMin); bn2[3] = -2*bd*((θMax + θMin)/2f0)/(θMax - θMin) ;
An2[4,4] = 2*bd/(qMax - qMin); bn2[4] = -2*bd*((qMax + qMin)/2f0)/(qMax - qMin) ;

An2Inv = inv(An2);

## =========
# Normalize perturbations, not states
bd = 5.0f0;
vdMin = -300f0; vdMax = 300f0;
αdMin = -deg2rad(30f0); αdMax = deg2rad(30f0);
θdMin = -deg2rad(30f0); θdMax = deg2rad(30f0);
qdMin = -deg2rad(30f0); qdMax = deg2rad(30f0);

vN3(v) = 2*bd*(v - (vdMax+vdMin)/2f0)/(vdMax - vdMin)
alpN3(α) = 2*bd*(α - (αdMax + αdMin)/2f0)/(αdMax - αdMin)
thN3(θ) = 2*bd*(θ - (θdMax + θdMin)/2f0)/(θdMax - θdMin)
qN3(q) = 2*bd*(q - (qdMax + qdMin)/2f0)/(qdMax - qdMin)

An3 = 1.0f0I(4);
bn3 = zeros(Float32,4);

An3[1,1] = 2*bd/(vdMax - vdMin); bn3[1] = -2*bd*((vdMax+vdMin)/2f0)/(vdMax - vdMin);
An3[2,2] = 2*bd/(αdMax - αdMin); bn3[2] = -2*bd*((αdMax + αdMin)/2f0)/(αdMax - αdMin);
An3[3,3] = 2*bd/(θdMax - θdMin); bn3[3] = -2*bd*((θdMax + θdMin)/2f0)/(θdMax - θdMin) ;
An3[4,4] = 2*bd/(qdMax - qdMin); bn3[4] = -2*bd*((qdMax + qdMin)/2f0)/(qdMax - qdMin) ;

An3Inv = inv(An3);

Kc_nomStab = Float32.([-1.09602  18.379   19.0464  9.91444
-1.11965  21.9143  21.942   9.77763]); # nominal stabilizing controller about linear perturbation plant

Kc_lqr = Float32.([ -0.00891916  0.0850919   0.568954    0.712423
-1.56943e-6  3.22057e-5  4.64721e-5  5.06978e-5]); # lqr controller with better behavior, inputs constrained. 

## ==== Trim values Normalized
xnTrim = An2*f18_xTrim[indX] + bn2
# xOrig = An\(xn - bn)

## ====== HOW YOU WANT NORMALIZED DYNAMICS TO LOOK
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

