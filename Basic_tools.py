#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as N

###################################
# --   GENERAL INFORMATIONS    -- #
###################################

_d2r = N.pi/180    # conversion factor from degrees to radians
# -- Coords in b = [-180,180]; l = [-90,90]
South_lb = [60,-30]
North_lb = [-120,30]

def Project_velocity_to_coords(l_object,b_object,
                               l_v,b_v,A_v):
    """
    l_v,b_v,A_v = coords and amplitude of the velocity vector.
    l_object,b_object =  coords of the object for which you want the projected velocity
    ----- 
    return amplitude of the projected velocity (COULD BE NEGATIVE)
    """
    return A_v * N.cos(ang_sep(l_object,b_object,l_v,b_v,
                               in_radian=True))

    
def ang_sep(l1,b1,l2,b2,in_radian=False):
    """
    Angular separation between two positions on the sky 
    (l1,b1) and (l2,b2) in degrees.
    """
    cos_theta = (N.cos(b1 * _d2r) * N.cos(b2 * _d2r) *
                 N.cos((l1 - l2) * _d2r) +
                 N.sin(b1 * _d2r) * N.sin(b2 * _d2r))
    ang_degree = N.arccos(cos_theta) / _d2r
    if in_radian:
        return ang_degree*_d2r
    return ang_degree


def dipole_comp(l,b):
    """
    Dipole components

    Arguments:
    l,b -- angular coordinates in degrees 
    """
    out = (N.cos(b*_d2r) * N.cos(l*_d2r),
           N.cos(b*_d2r) * N.sin(l*_d2r),
           N.sin(b*_d2r))
        
    return N.asarray(out)

def convert_spherical(Dipole_cartesian_velocity, cov=None):
    """
    Convert fit results in Cartesian coordinates to spherical coordinates 
    (angles in degrees). Covariance matrix can be converted as well
    if it is stated.
    """
    x = Dipole_cartesian_velocity[0]
    y = Dipole_cartesian_velocity[1] 
    z = Dipole_cartesian_velocity[2] 

    v = N.sqrt(x**2 + y**2 + z**2)
    v_sph = N.array([v, (N.arctan2(y,x) / _d2r + 180) % 360 - 180, 
                          N.arcsin(z/v) / _d2r])
    
    if cov is None:
        return v_sph
    else:
        cov_out = deepcopy(cov)    

        jacobian = N.zeros((3,3))
        jacobian[0,0] = x / v
        jacobian[1,0] = - y / (x**2 + y**2)
        jacobian[2,0] = - x * z / (v**2 * N.sqrt(x**2 + y**2))
        jacobian[0,1] = y / v
        jacobian[1,1] = x / (x**2 + y**2)
        jacobian[2,1] = - y * z / (v**2 * N.sqrt(x**2 + y**2))
        jacobian[0,2] = z / v
        jacobian[1,2] = 0
        jacobian[2,2] = N.sqrt(x**2 + y**2) / (v**2)

        cov_sph = (jacobian.dot(cov_out)).dot(jacobian.T)
        cov_sph[1,1] /= _d2r**2
        cov_sph[2,2] /= _d2r**2
        cov_sph[2,1] /= _d2r**2
        cov_sph[1,2] /= _d2r**2
        cov_sph[0,1] /= _d2r
        cov_sph[0,2] /= _d2r
        cov_sph[1,0] /= _d2r
        cov_sph[2,0] /= _d2r    

        return v_sph, cov_sph


# -------------------------------- #
# ----  FROM THE SNf ToolBox ----- #
# -------------------------------- #
def radec2gcs(ra, dec, deg=True):
    """
    Authors: Yannick Copin (ycopin@ipnl.in2p3.fr)
    
    Convert *(ra,dec)* equatorial coordinates (J2000, in degrees if
    *deg*) to Galactic Coordinate System coordinates *(lII,bII)* (in
    degrees if *deg*).

    Sources:

    - http://www.dur.ac.uk/physics.astrolab/py_source/conv.py_source
    - Rotation matrix from
      http://www.astro.rug.nl/software/kapteyn/celestialbackground.html

    .. Note:: This routine is only roughly accurate, probably at the
              arcsec level, and therefore not to be used for
              astrometric purposes. For most accurate conversion, use
              dedicated `kapteyn.celestial.sky2sky` routine.

    >>> radec2gal(123.456, 12.3456)
    (210.82842704243518, 23.787110745502183)
    """

    if deg:
        ra  =  ra * _d2r
        dec = dec * _d2r

    rmat = N.array([[-0.054875539396, -0.873437104728, -0.48383499177 ],
                    [ 0.494109453628, -0.444829594298,  0.7469822487  ],
                    [-0.867666135683, -0.198076389613,  0.455983794521]])
    cosd = N.cos(dec)
    v1 = N.array([N.cos(ra)*cosd,
                  N.sin(ra)*cosd,
                  N.sin(dec)])
    v2 = N.dot(rmat, v1)
    x,y,z = v2

    c,l = rec2pol(x,y)
    r,b = rec2pol(c,z)

    assert N.allclose(r,1), "Precision error"

    if deg:
        l /= _d2r
        b /= _d2r

    return l, b

def rec2pol(x,y, deg=False):
    """
    Authors: Yannick Copin (ycopin@ipnl.in2p3.fr)
    
    Conversion of rectangular *(x,y)* to polar *(r,theta)*
    coordinates
    """

    r = N.hypot(x,y)
    t = N.arctan2(y,x)
    if deg:
        t /= RAD2DEG

    return r,t
