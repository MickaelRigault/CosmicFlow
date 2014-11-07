#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as N

import Sample_tools as Sp
import Basic_tools as Bt
from iminuit import Minuit#, describe, Struct

File_union = "union2pos2_vcluster.dat"
File_snf   = "snf_ACEv3_vcluster.dat"

def Load_Union_Supernovae(load="union",remove_names=['']):
    """
    """
    # 0=SNname 1=l ,2=b , 3=redshift, 4=mu, 5=dmu
    #Input Supernova = coords,mu,redshift,dmu,dredshift,HR_given=False,SNname="NoName",

    if load.lower() == "union" :
        File_to_load = File_union
    elif load.lower() == "snf" :
        File_to_load = File_snf
        
    
    Supernovae = []
    for line in open(File_to_load).read().splitlines():
        if line[0] == "#":
            continue
        SNname,ra,dec,zcmb,mu,dmu = line.split()
        
        if SNname.lower() in N.asarray([l.lower() for l in remove_names]):
            continue
        
        coords_lb = Bt.radec2gcs(N.float(ra),N.float(dec))
        Supernovae.append( Sp.Supernova( coords_lb, N.float(mu),N.float(zcmb),N.float(dmu),
                                      HR_given=False,SNname=SNname))
        
    return Supernovae

        

def Load_Union_Cosmic_flow(**kwargs):
    """
    """
    return Cosmic_flow(Load_Union_Supernovae(**kwargs))
    
    

#########################################################
#  --- THE MAIN CLASS - Supernova Class as Inputs  ---  #
#########################################################
class Cosmic_flow( Sp.Sample ):
    """
    """
    
    def Select_Supernovae_to_Fit(self,redshift_range=[0,2],intrinsic_to_remove=0.):
        """
        """
        self.Supernovae_to_Fit =[SN for SN in self.Supernovae
                                 if SN.z>redshift_range[0] and SN.z<redshift_range[1] and
                                 SN.dmu>intrinsic_to_remove]
        
        self.mu_to_fit  = N.asarray([SN.mu for SN in self.Supernovae_to_Fit
                                    ])
        self.dmu_to_fit = N.asarray([N.sqrt(SN.dmu**2-intrinsic_to_remove**2) for SN in self.Supernovae_to_Fit
                                     ])
        self.N_SNe_fitted = len(self.mu_to_fit)
        
    # ----------------------------- #
    # ---  FIT COSMIC FLOW      --- #
    # ----------------------------- #
    # -- deltaZ, l,b
    def _read_guess_(self,Pguess,Boundaries):
        """
        """
        if self.Flow_fit == "dipole":
            if Pguess is None or len(Pguess) != 3:
                self.Pguess = [200,30,300]
            else:
                self.Pguess = Pguess
                
            if Boundaries is None or N.shape(Boundaries) != (3,2):
                self.Boundaries = [[-180,180],[-90,90],[0,None]]
            else:
                self.Boundaries = Boundaries
                
    def _read_parameters_(self,l_v,b_v,Ampl_v):
        """
        THIS IS MODEL DEPENDENT
        """
        self._current_dipole_parameter_ = [l_v,b_v,Ampl_v]
        self._current_cosmo_FlowMu_ = N.asarray([SN.Cosmo_DistMu_given_Dipole(self._current_dipole_parameter_)
                                                 for SN in self.Supernovae_to_Fit])
    
    def _chi2_(self,l_v,b_v,Ampl_v):
        """
        """
        #if self.Flow_fit = "dipole":
        self._read_parameters_(l_v,b_v,Ampl_v) # Flow dependent

        return N.sum((self.mu_to_fit - self._current_cosmo_FlowMu_)**2/self.dmu_to_fit**2 )
        

    def _setup_minuit_(self,Flow_fit,Pguess=None,Boundaries=None):
        """
        """
        self.Flow_fit = "dipole"
        self._read_guess_(Pguess,Boundaries)
        
        if Flow_fit.lower() == "dipole":
            self.minuit = Minuit(self._chi2_,
                                l_v =   self.Pguess[0],       limit_l_v    = self.Boundaries[0], error_l_v=30.,
                                b_v =   self.Pguess[1],       limit_b_v    = self.Boundaries[1],error_b_v=30.,
                                Ampl_v= self.Pguess[2],  limit_Ampl_v = self.Boundaries[2],error_Ampl_v=100., 
                                print_level=1, errordef=1)
        else:
            self.Flow_fit = "FAILURE"
            raise ValueError("ONLY DIPOLE IMPLEMENTED YET")
        

        
    def Fit_Flow(self,Pguess=None,Dipole=True,verbose=True):
        """
        """
        if "Supernovae_to_Fit" not in dir(self):
            raise ValueError("You should first Select the SNe you want to fit with the Select_Supernovae_to_Fit function")
    
        self._setup_minuit_(Flow_fit="dipole",Pguess=Pguess)
        self._fit_output_ = self.minuit.migrad()
        if self._fit_output_[0]["is_valid"] is False:
            print "** WARNING ** migrad is not valid -> Minuit FAILED on fitting  the Dipole ! "
            
        if verbose:
            self.Print_results()
            
    def Print_results(self):
        """
        """
        if "_fit_output_" not in dir(self):
            raise ValueError("Please First run Fit_Flow")
        
        print " Cosmic flow fit output ".center(30,"*")
        if self.Flow_fit == "dipole":
            print " for a fit of %d Supernovae "%self.N_SNe_fitted
            print ("Dipole amplitude %.1f"%self.minuit.values["Ampl_v"]).center(30," ")
            print ("Dipole (l,b) = %.1f,%.1f"%(self.minuit.values["l_v"],self.minuit.values["b_v"])).center(30," ")
            print "".center(30,"*")
            
