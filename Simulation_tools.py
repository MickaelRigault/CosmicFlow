#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as N
import Cosmo_basics as Cb
import Basic_tools  as Bt

import Sample_tools as St



def simulate_l_b_coverage(Npoints,coverage="full_sky",
                          pole_sky_exclision=50.):
    """
    """
    # ----------------------- #
    # --                   -- #
    # ----------------------- #
    def _draw_lb_(Npoints_):
        """
        """
        l = N.random.random(Npoints_)*360. - 180
        b = N.arcsin(N.random.random(Npoints_)*2. - 1) / Bt._d2r
        if Npoints_==1:
            return l[0],b[0]
        return l,b

    def _hemisphere_sky_(To_exclude_coords_lb):
        """
        """
        l,b = [],[]
        while( len(l)< Npoints ):
            l_,b_ = _draw_lb_(1)
            if Bt.ang_sep(l_,b_,To_exclude_coords_lb[0],To_exclude_coords_lb[1],in_radian=False)>pole_sky_exclision:
                l.append(l_)
                b.append(b_)
        return l,b

    # ----------------------- #
    # --                   -- #
    # ----------------------- #
    
    if coverage.lower() == "full_sky":
        # -- ALL SKY -- #
        return _draw_lb_(Npoints)
    
    elif coverage.lower() == "northern_sky":
        # -- NORTHEN SKY -- #
        return _hemisphere_sky_(Bt.South_lb)
    
    elif coverage.lower() == "southern_sky":
        # -- SOUTHERN SKY -- #
        return _hemisphere_sky_(Bt.North_lb)
    
    else:
        raise ValueError("Only 'full_sky'/'northern_sky'/'southern_sky' coverage have been implemented (%s requested)."%coverage)



def simulate_Dipole_mu(redshits,coords,Dipole,
                       Normal_measured_error=[0.10,0.02],
                       intrinsic_dispersion=0.1,**cosmo):
    """
    Errors will be drawn from a normal distribution Normal_measured_error = [mean, sigma]
    In addition, The SNe Ia will be dispersed by a gaussian intrinsic_dispersion error
    """
    cosmo = Cb.Cosmology(**cosmo)
    # --- Dipole Parameters --- #
    l_dipole,b_dipole,Ampl_dipole = Dipole
    mu_sim,dmu_sim = [],[]
    for i,z in enumerate(redshits):
        l_i,b_i   = coords[i]
        Dip_v_i   = Bt.Project_velocity_to_coords(l_i,b_i,l_dipole,b_dipole,Ampl_dipole)
        
        #  -- Prediction -- #
        mu_cosmo_dipole = cosmo.Mu(z + Dip_v_i / Cb.CLIGHTkm)
        
        #  --   Noise    -- #
        if Normal_measured_error is None or len(Normal_measured_error) != 2:
            
            mu_err_measured = 0
        else:
            mu_err_measured = N.random.normal(Normal_measured_error[0],Normal_measured_error[1])
            
        dmu = N.sqrt( mu_err_measured**2 + intrinsic_dispersion**2)
        # -- The "observed" mu include these (gaussian) errors -- #
        if dmu>0:
            
            mu = mu_cosmo_dipole +  N.random.normal(0,dmu)
        else:
            mu = mu_cosmo_dipole
        
        ## ---> Simulation
        mu_sim.append(mu)
        dmu_sim.append(dmu)

    return N.asarray(mu_sim),N.asarray(dmu_sim)
    
    

    
def Simulate_Cosmic_flow_Supernovae(redshift_range,Dipole=[0,0,1000],Npoints=1000,
                                coverage="full_sky",**kwargs):
    """
    redshift_range = [zmin,zmax]
    Dipole=[l_dipole,b_dipole,amplitude_Dipole_in_km/s]

    kwargs goes to simulate_Dipole_mu. Include Errors assumed and **cosmo
    """
    if len(redshift_range) != 2:
        raise ValueError("redshift_range must be like [z_min, z_max] ")
    
    redshifts   = N.random.random(Npoints)*(N.max(redshift_range) - N.min(redshift_range)) + N.min(redshift_range)

    l_sim,b_sim = simulate_l_b_coverage(Npoints,coverage)
    Coords_sim = N.asarray([l_sim,b_sim]).T
    mu_simulate,dmu_simulate = simulate_Dipole_mu(redshifts, Coords_sim, Dipole,**kwargs)
    
    # Supernova input: coords,mu,redshift,dmu,dredshift=None,HR_given=False,
    return [St.Supernova( Coords_sim[i],  mu_simulate[i], redshifts[i], dmu_simulate[i])
            for i in range(Npoints)]
    
    
