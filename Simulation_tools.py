#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as N
import Cosmo_basics as Cb
import Basic_tools  as Bt

import Sample_tools as St


def simulate_l_b_coverage(Npoints,MW_exclusion=10,ra_range=(-180,180),dec_range=(-90,90),
                          output_frame='galactic'):
    """
    """
    # ----------------------- #
    # --                   -- #
    # ----------------------- #
    def _draw_radec_(Npoints_,ra_range_,dec_sin_range_):
        """
        """
        ra = N.random.random(Npoints_)*(ra_range_[1] - ra_range_[0]) + ra_range_[0]
        dec = N.arcsin(N.random.random(Npoints_)*(dec_sin_range_[1] - dec_sin_range_[0]) + dec_sin_range_[0]) / Bt._d2r

        return ra,dec

    def _draw_without_MW_(Npoints_,ra_range_,dec_sin_range_,MW_exclusion_):
        """
        """
        
        l,b = N.array([]),N.array([])
        while( len(l) < Npoints_ ):
            ra,dec = _draw_radec_(Npoints_ - len(l),ra_range_,dec_sin_range_)
            l_,b_ = Bt.radec2gcs(ra,dec)
            if output_frame == 'galactic':
                l = N.concatenate((l,l_[N.abs(b_)>MW_exclusion_]))
                b = N.concatenate((b,b_[N.abs(b_)>MW_exclusion_]))
            else:
                l = N.concatenate((l,ra[N.abs(b_)>MW_exclusion_]))
                b = N.concatenate((b,dec[N.abs(b_)>MW_exclusion_]))                

        return l,b

    # ----------------------- #
    # --                   -- #
    # ----------------------- #

    if output_frame not in ['galactic','j2000']:
        raise ValueError('output_frame must "galactic" or "j2000"')

    if ra_range[0] < -180 or ra_range[1] > 180 or ra_range[0] > ra_range[1]:
        raise ValueError('ra_range must be contained in [-180,180]')

    if dec_range[0] < -90 or dec_range[1] > 90 or dec_range[0] > dec_range[1]:
        raise ValueError('dec_range must be contained in [-180,180]')

    dec_sin_range = (N.sin(dec_range[0]*Bt._d2r),N.sin(dec_range[1]*Bt._d2r)) 

    if MW_exclusion > 0.:
        return _draw_without_MW_(Npoints,ra_range,dec_sin_range,MW_exclusion)
    else:
        ra,dec = _draw_radec_(Npoints,ra_range,dec_sin_range)
        if output_frame == 'galactic':
            return Bt.radec2gcs(ra,dec)
        else:
            return ra,dec



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
                                    ra_range=(-180,180),dec_range=(-90,90),MW_exclusion=10,
                                    **kwargs):
    """
    redshift_range = [zmin,zmax]
    Dipole=[l_dipole,b_dipole,amplitude_Dipole_in_km/s]

    kwargs goes to simulate_Dipole_mu. Include Errors assumed and **cosmo
    """
    if len(redshift_range) != 2:
        raise ValueError("redshift_range must be like [z_min, z_max] ")
    
    redshifts   = N.random.random(Npoints)*(N.max(redshift_range) - N.min(redshift_range)) + N.min(redshift_range)

    l_sim,b_sim = simulate_l_b_coverage(Npoints,MW_exclusion,ra_range,dec_range)
    Coords_sim = N.asarray([l_sim,b_sim]).T
    mu_simulate,dmu_simulate = simulate_Dipole_mu(redshifts, Coords_sim, Dipole,**kwargs)
    
    # Supernova input: coords,mu,redshift,dmu,dredshift=None,HR_given=False,
    return [St.Supernova( Coords_sim[i],  mu_simulate[i], redshifts[i], dmu_simulate[i])
            for i in range(Npoints)]
    
    
