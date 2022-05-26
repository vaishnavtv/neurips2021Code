f18_xTrim = Float32.(1.0e+02*[3.500000000000000;-0.000000000000000;0.003540971009312;-0.000189987747098;0.000322083113778;0.000459982356948;0.006108652381980;0.003262466855376;0;]);
indX = [1;3;3;5];

f18_uTrim = Float32.(1.0e+04 *[-0.000000767698465;-0.000002371697733;-0.000007859275313;1.449999997030301;]);
indU = [3;4];

function f18Dyn(x,u)
    
        S = 400f0;                # Reference Area, ft^2
        b =  37.42f0;             # Wing Span, ft
        c =  11.52f0;             # Aerodynamic Mean Chord, ft
        rho = 1.0660f-003;      # Air Density, slugs/ft^3  --- 25C / 25000 ft
        Ixx = 23000f0;            # Principle Moment of Intertia around X-axis, slugs*ft^2
        Iyy = 151293f0;           # Principle Moment of Intertia around Y-axis,slugs*ft^2
        Izz = 169945f0;           # Principle Moment of Intertia around Z-axis,slugs*ft^2
        Ixz = -2971f0;            # Principle Moment of Intertia around XZ-axis,slugs*ft^2
        m = 1034.5f0;             # mass, slugs
        g = 32.2f0;               # gravitational constant,ft/s^2
    
        d2r = Float32(pi/180);
        r2d = Float32(1/d2r);
    
        Clb_0 = -0.0556f0;
        Clb_1 = -0.4153f0;
        Clb_2 = -0.3620f0;
        Clb_3 = 2.3843f0;
        Clb_4 = -1.6196f0;
    
        Cldr_0= 0.0129f0;
        Cldr_1= 0.0014f0;
        Cldr_2= 0.0083f0;
        Cldr_3= -0.0274f0;
    
        Clda_0= 0.1424f0;
        Clda_1= -0.0516f0;
        Clda_2= -0.2646f0;
        Clda_3= 0.1989f0;
    
        Clp_0= -0.3540f0;
        Clp_1= 0.2377f0;
    
        Clr_0= 0.1983f0;
        Clr_1= 0.7804f0;
        Clr_2= -1.0871f0;
    
        Cnb_0= 0.0885f0;
        Cnb_1= 0.0329f0;
        Cnb_2= -0.3816f0;
    
        Cndr_0= -0.0780f0;
        Cndr_1= -0.0176f0;
        Cndr_2= 0.5564f0;
        Cndr_3= -0.8980f0;
        Cndr_4= 0.3899f0;
    
        Cnda_0= 0.0104f0;
        Cnda_1= 0.0584f0;
        Cnda_2= -0.3413f0;
        Cnda_3= 0.2694f0;
    
        Cnr_0= -0.4326f0;
        Cnr_1= -0.1307f0;
    
        Cnp_0= 0.0792f0;
        Cnp_1= -0.0881f0;
    
        Cyb_0= -0.7344f0;
        Cyb_1= 0.2654f0;
        Cyb_2= -0.1926f0;
    
        Cyda_0= -0.1656f0;
        Cyda_1= -0.2403f0;
        Cyda_2= 1.5317f0;
        Cyda_3= -0.8500f0;
    
        Cydr_0= 0.2054f0;
        Cydr_1= 0.4082f0;
        Cydr_2= -1.6921f0;
        Cydr_3= 0.9351f0;
    
        Cma_0= -0.0866f0;
        Cma_1= 0.5110f0;
        Cma_2= -1.2897f0;
    
        Cmds_0= -0.9051f0;
        Cmds_1= -0.3245f0;
        Cmds_2= 0.9338f0;
    
        Cmq_0= -4.1186f0;
        Cmq_1= 10.9921f0;
        Cmq_2= -68.5641f0;
        Cmq_3= 64.7190f0;
    
        CLds_0= 0.5725f0;
        CLds_1= 0.4055f0;
        CLds_2= -2.6975f0;
        CLds_3= 2.1852f0;
    
        Cdds_0= 0.0366f0;
        Cdds_1= -0.2739f0;
        Cdds_2= 4.2360f0;
        Cdds_3= -3.8578f0;
    
    
        # ==========================================================================
        # State ordering
    
    # get longitudinal modes
    # v, alpha, theta, q (rest = 0) : 4 states
    # theta = alpha, q = 0: trim not required as nonlinear control generated
    # operating point: use steady level flight from trimf18.m
    # nonlinear dynamics about trim in terms of perturbation 
    # wrapper around this function to set rest state  = 0 (other values frozen)
    # thrust and d_STAB for inputs
    
        V       =  x[1];       # Airspeed , ft/s
        beta    =  x[2];       # Sideslip Angle, rad
        alpha   =  x[3];       # Angle-of-attack, rad
    
        p       =  x[4];       # Roll rate, rad/s
        q       =  x[5];       # Pitch rate, rad/s
        r       =  x[6];       # Yaw rate, rad/s
    
        phi     =  x[7];       # Roll Angle, rad
        theta   =  x[8];       # Pitch Angle, rad
        psi     =  x[9];       # Yaw Angle, rad
    
    
        # Input Terms
        d_STAB   = u[3];       # Stabilator Deflection, rad
        d_RUD    = u[2];       # Rudder Deflection, rad
        d_AIL    = u[1];       # Aileron Deflection, rad
        T        = u[4];       # Thrust
    
        cosbeta = cos(beta);
        cos2beta3 = cos(2*beta/3);
        sinbeta = sin(beta);
        tanbeta = tan(beta);
        secbeta = sec(beta);
    
        cosalpha = cos(alpha);
        sinalpha = sin(alpha);
    
        cosphi = cos(phi);
        sinphi = sin(phi);
    
        costheta = cos(theta);
        sintheta = sin(theta);
        sectheta = sec(theta);
        tantheta =  tan(theta);
    
        # ==========================================================================
        # Aerodynamic Model
    
        # # -------------------------------------------------
        # Rolling Moment
        Clb     =  Clb_0 + Clb_1*alpha + Clb_2*alpha^2 + Clb_3*alpha^3  + Clb_4*alpha^4;
        Cldr    =  Cldr_0 + Cldr_1*alpha + Cldr_2*alpha^2 + Cldr_3*alpha^3;
        Clda    =  Clda_0 + Clda_1*alpha + Clda_2*alpha^2 + Clda_3*alpha^3 ;
        Clp     =  Clp_0 + Clp_1*alpha;
        Clr     =  Clr_0 + Clr_1*alpha + Clr_2*alpha^2;
    
        # ----- Total Rolling Moment
        C_l     =  Clb*beta + Clda* d_AIL + Cldr*d_RUD + Clp*p*b/2/V + Clr*r*b/2/V;
    
        # -------------------------------------------------
        #  Yawing Moment
        Cnb     = Cnb_0 + Cnb_1*alpha + Cnb_2*alpha^2 ;
        Cndr    = Cndr_0 + Cndr_1*alpha + Cndr_2*alpha^2 + Cndr_3*alpha^3 + Cndr_4*alpha^4;
        Cnda    = Cnda_0 + Cnda_1*alpha + Cnda_2*alpha^2 + Cnda_3*alpha^3 ;
        Cnr     = Cnr_0 + Cnr_1*alpha;
        Cnp     = Cnp_0 + Cnp_1*alpha;
    
        # ----- Total Yawing Moment
        C_n     = Cnb*beta + Cnda*d_AIL + Cndr*d_RUD + Cnr*r*b/2/V + Cnp*p*b/2/V;
    
        # -------------------------------------------------
        # SideForce
    
        Cyb     = Cyb_0 + Cyb_1*alpha + Cyb_2*alpha^2 ;
        Cyda    = Cyda_0 + Cyda_1*alpha + Cyda_2*alpha^2 + Cyda_3*alpha^3;
        Cydr    = Cydr_0 + Cydr_1*alpha + Cydr_2*alpha^2 + Cydr_3*alpha^3;
    
        # -------- Total Side Force
        C_Y     = Cyb*beta + Cydr*d_RUD +  Cyda*d_AIL;
    
        # -------------------------------------------------
        # Pitching Moment
        Cma     =  Cma_0 + Cma_1*alpha + Cma_2*alpha^2;
        Cmds    =  Cmds_0 + Cmds_1*alpha + Cmds_2*alpha^2;
        Cmq     =  Cmq_0 + Cmq_1*alpha + Cmq_2*alpha^2 + Cmq_3*alpha^3 ;
    
        # --- Total Pitching Moment
        C_m     =  Cma + Cmds* d_STAB  +  Cmq*c*q/2/V;
    
        # -------------------------------------------------
        # Lift Coefficient
        CLds = CLds_0 + CLds_1*alpha+ CLds_2*alpha^2 + CLds_3*alpha^3;
    
        C_lift = (-0.0204f0 + 5.677f0*alpha - 5.4246f0*alpha^2 + 1.1645f0*alpha^3)*cos2beta3 +  CLds*d_STAB;
    
    
        # -------------------------------------------------
        # Drag Coefficient
        Cdds = Cdds_0 + Cdds_1*alpha+ Cdds_2*alpha^2 + Cdds_3*alpha^3;
    
        C_drag =  (-1.4994f0 - 0.1995f0*alpha + 6.3971f0*alpha^2 - 5.7341f0*alpha^3 + 1.4610f0*alpha^4) *cosbeta + 1.5036f0 + Cdds*d_STAB ;
    
        # -------------------------------------------------
        # Form Aerodynamic forces and moments
    
        qbar = 1f0/2f0*rho*V^2;  # Dynamic pressure
        L =  C_l*qbar*S*b ;
        M =  C_m*qbar*S*c;
        N =  C_n*qbar*S*b;
        Y =  C_Y*qbar*S ;
        Lift = C_lift*qbar*S ;
        Drag = C_drag*qbar*S ;
    
        # Body to Wind Axis Conversion of the Aerdynamic data
        CD_w = C_drag*cosbeta - C_Y*sinbeta;
        CY_w =  C_Y*cosbeta + C_drag*sinbeta;
    
        # --------------------------------------------------------------------------
        # Force Equation
    
        Vd =  -qbar*S*CD_w /m + g*(cosphi*costheta*sinalpha*cosbeta  + sinphi*costheta*sinbeta -sintheta*cosalpha*cosbeta) + (T/m)*cosbeta*cosalpha;
    
        betad =  qbar*S*CY_w/m/V + p*sinalpha - r*cosalpha + g*costheta*sinphi*cosbeta/V + sinbeta*(g*cosalpha*sintheta - g*sinalpha*cosphi*costheta + T*cosalpha/m)/V;
    
        alphad = -Lift*secbeta/m/V + q - tanbeta*(p*cosalpha + r*sinalpha) + g*(cosphi*costheta*cosalpha + sintheta*sinalpha)*secbeta/V -(T*secbeta/m/V)*sinalpha;
    
    
        #--------------------------------------------------------------------------
        #Moment Equations
    
        pd = ((Izz* L + Ixz* N - (Ixz*(Iyy-Ixx-Izz)*p + (Ixz^2 + Izz*(Izz - Iyy))*r)*q)/(Ixx*Izz -Ixz^2));
    
        qd = ((M + (Izz -Ixx)*p*r + (r^2 -p^2)*Ixz)/Iyy);
    
        rd =  ((Ixz * L  + Ixx * N + (Ixz * (Iyy - Ixx -Izz) *r + (Ixz^2 + Ixx*(Ixx - Iyy))*p)*q)/(Ixx*Izz -Ixz^2));
    
        #--------------------------------------------------------------------------
        # Kinetics Equation
    
        phid = p + (q*sinphi + r*cosphi)*tantheta;
        thetad = q*cosphi - r*sinphi;
        psid =  (q*sinphi + r*cosphi)*sectheta;
    
        #--------------------------------------------------------------------------
        # Group state derviatives
    
        yd = [Vd ;betad; alphad; pd; qd; rd; phid; thetad; psid];
        return yd
    end