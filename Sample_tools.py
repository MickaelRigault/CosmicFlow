#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy       as N
from scipy import percentile
import Cosmo_basics as Cb
import Basic_tools  as Bt
import matplotlib.pyplot as P    
    


#########################################################
#  ----- THE SAMPLE OBJECT                      -----   #
#########################################################
class Sample( object ):
    """
    """
    # ----------------------------- #
    # ---  INITIATE THE CLASS   --- #
    # ----------------------------- #
    def __init__(self,Supernovae,load_deltaZ=False,**cosmo):
        """
        **cosmo is a kwargs that goes in Cosmo_basics.Cosmology, (DEFAULT: h=0.71, Om=0.27, Ol=0.73, w=-1, ref='LCDM')
        """
        self._Load_Sample_(Supernovae,**cosmo)
        if load_deltaZ:
            self._load_deltaZ_()
        
    def _Load_Sample_(self,Supernovae,**cosmo):
        """
        """
        self.Supernovae = Supernovae
        # -- Assumed Cosmology -- #
        self._load_Cosmo_(**cosmo)
        
        # -- Observed mu -- #
        self.mu  =  N.asarray([ SN.mu for SN in self.Supernovae])
        self.dmu =  N.asarray([SN.dmu for SN in self.Supernovae])
        self.HR  =  N.asarray([ SN.HR for SN in self.Supernovae])
        self.dHR =  N.asarray([SN.dHR for SN in self.Supernovae])
        self.HRsig =  self.HR/self.dHR
        # -- Observed redshift -- #
        self.z   =  N.asarray([ SN.z for SN in self.Supernovae])
        self.dz  =  N.asarray([SN.dz for SN in self.Supernovae])
        # -- Observed Coordinate -- #
        self.l   =  N.asarray([ SN.l for SN in self.Supernovae])
        self.b   =  N.asarray([ SN.b for SN in self.Supernovae])
        
        # -- Observed mu -- #
        self.sample_size= len(self.mu)

    # ----------------------------- #
    # ---  LOADING TOOLS        --- #
    # ----------------------------- #
    def _load_Cosmo_(self,**cosmo):
        """
        """
        self.cosmo = Cb.Cosmology(**cosmo)
        [SN._load_cosmo_(**cosmo) for SN in self.Supernovae]
         
    def _load_deltaZ_(self,Verbose=0):
        """
        """
        [SN.Fit_delta_Z(print_level=Verbose) for SN in self.Supernovae]
        self.deltaZ = N.asarray([SN.deltaZ   for SN in self.Supernovae])
        self.ddeltaZ = N.asarray([SN.ddeltaZ for SN in self.Supernovae])
        self.deltaV = self.deltaZ * Cb.CLIGHTkm
    def Change_Cosmology(self,**new_cosmo):
        """
        """
        self._load_Cosmo_(**new_cosmo)
        self._load_deltaZ_()
        

    def Load_Subsample(self,redshift_range=[0,2]):
        """
        """
        self.SubSample = Sample( [SN for SN in self.Supernovae
                                  if SN.z>redshift_range[0] and SN.z<redshift_range[1]])
        
    def Load_Plot(self,show_subsample=False):
        """
        """
        if show_subsample:
            self.Plot = Sample_plot(self.SubSample)
        else:
            self.Plot = Sample_plot(self)
        
        

#########################################################
#  ----- THE SUPERNOVAE OBJECT                  -----   #
#########################################################
class Supernova( object ):
    """
    """
    def __init__(self,coords,mu,redshift,dmu,dredshift=None,HR_given=False,
                 SNname="NoName",
                 **cosmo):
        """
        """
        self.l = coords[0]
        self.b = coords[1]
        self.z  =  redshift
        self.dz = dredshift
        
        self._load_cosmo_(**cosmo)
        self._load_mu_(mu,dmu,HR_given)

        self.object = SNname
        
        

    def _load_mu_(self,mu,dmu,HR_given):
        """
        This assumes dmu already account for the error on z.
        No intrinsic dispersion.
        """
        if HR_given:
            self.HR  =  mu
            self.dHR = dmu
            self.mu  = self.cosmo.Mu(self.z) + self.HR
            self.dmu = self.dHR
        else:
            self.mu  =  mu
            self.dmu = dmu
            self.HR  = self._load_HR_(update=False)
            self.dHR = dmu
            
            
    def _load_cosmo_(self,**cosmo):
        """
        **cosmo is a kwargs that goes in Cosmo_basics.Cosmology, (DEFAULT: h=0.71, Om=0.27, Ol=0.73, w=-1, ref='LCDM')
        """
        self.cosmo = Cb.Cosmology(**cosmo)

            
    def _load_HR_(self,delta_z=0,update=True):
        """
        With delta z you can change the observed self.z, such as self.z = self.z+delta_z
        """
        
        if "cosmo" not in dir(self) or self.cosmo is None:
            raise ValueError("Sorry I can't load the Hubble residuals without self.cosmo, run _load_cosmo_")
        if update:
            self._current_HR_  = self.mu - self.cosmo.Mu(self.z+delta_z)
            self._current_dHR_ = self.dHR
        else:
            return self.mu - self.cosmo.Mu(self.z+delta_z)
        
        
    ###################################
    #   ----  FIND DELTA Z ------     # 
    ###################################
    def _toFit_toGet_deltaZ_(self,deltaZ):
        """
        """
        self._load_HR_(deltaZ)
        return (self._current_HR_/self._current_dHR_)**2
        
    def _setup_minuit_(self,print_level=1):
        """
        """
        from iminuit import Minuit#, describe, Struct
        self.minuit = Minuit(self._toFit_toGet_deltaZ_,
                                    deltaZ= 0, error_deltaZ=1e-7, limit_deltaZ = [-0.01,0.01],
                                    print_level=print_level,errordef=1)
    def Fit_delta_Z(self,**kwargs):
        """
        **kwargs goes to _setup_minuit_
        """
        self._setup_minuit_(**kwargs)
        
            
        self._fit_output_ = self.minuit.migrad()
        if self._fit_output_[0]["is_valid"] is False:
            print "** WARNING ** migrad is not valid -> Minuit fit of the delta_Z failed ! (SN: %s)"%self.object

        self.deltaZ  = self.minuit.values["deltaZ"]
        self.ddeltaZ = self.minuit.errors["deltaZ"]


        
    def Dipole_velocities_to_Delta_z(self,Dipole_param):
        """
        THIS IS THE FUNCTION TO BE GENERALIZED
        -------
        return the projected radial velocity [in km/s] of the dipole at the SN location (0 is angle sep of 90degree)
        """
        if len(Dipole_param) != 3:
            raise ValueError('Dipole_param must be l_v,b_v,Ampl_v for Now')

        l_v,b_v,Ampl_v = Dipole_param
        return Bt.Project_velocity_to_coords(self.l,self.b,l_v,b_v,Ampl_v) / Cb.CLIGHTkm

    
    def Cosmo_DistMu_given_Dipole(self,Dipole_param):
        """
        """
        DeltaZ = self.Dipole_velocities_to_Delta_z(Dipole_param)
        
        return self.cosmo.Mu(self.z+DeltaZ)


#########################################################
#  ----- SAMPLE PLOT CLASS                      -----   #
#########################################################
class Sample_plot( object ):
    """
    """
    def __init__(self,Sample_class,scatter_cmap=P.cm.coolwarm):
        """
        """
        self.Samp = Sample_class
        self.scatter_cmap = scatter_cmap

        self.default_skyplot_kwargs = dict(marker="o",mew=0,ms=15)
        
    def SkyPlot(self,axin=None,savefile=None,colored_by=None,
                vmin=0,vmax=1,
                **kwargs):
        """
        """
        self._setup_skyaxes_(axin=axin)
        self._load_sky_scatter_color_(colored_by,vmin=vmin,vmax=vmax)

        kwargs_ = self.default_skyplot_kwargs.copy()
        for k in kwargs.keys():
            kwargs_[k] = kwargs[k]
            
        # -- as.scatter crashes in Mac OS X -- #
        [self.ax.plot(self.Samp.l[i]*Bt._d2r,self.Samp.b[i]*Bt._d2r,c=self._color_used_[i],
                      **kwargs_)
         for i in range(len(self.Samp.l))]
        self.ax.grid(True)
        self._add_colorbar_()
        self._readout_(savefile)



    def Hubble_Residual_Plots(self,axin=None,savefile=None):
        """
        """
        
    ######################
    # -- Axis Setups --- #
    ######################
    def _setup_skyaxes_(self,axin,add_colorbar=False):
        """
        """
        if axin is None:
            self.fig = P.figure(figsize=[10,5])
            self.ax  = self.fig.add_subplot(111, projection="mollweide", axisbg ='w')
        else:
            self.ax = axin
            self.fig = self.ax.figure

    def _add_colorbar_(self,axcolorbar=None,
                       label= None,
                       verticale=True,
                       no_ticks=True,
                       add_legend=True):
        """
        """
        if axcolorbar is None:
            if verticale:
                self.axcbar = self.fig.add_axes([0.9,0.10,0.03,0.80])
            else:
                self.axcbar = self.fig.add_axes([0.10,0.9,0.80,0.04])
        else:
            self.axcbar = axcolorbar

        # ----------------- #
        vmin,vmax = self._skyPlot_color_ranges_
        norm = P.matplotlib.colors.Normalize(vmin=vmin,
                                            vmax=vmax)
        
        if verticale:
            x,y=N.mgrid[1:10:0.05,1:10]
            self._colorbar_ = self.axcbar.imshow(10-x, cmap=self.scatter_cmap)
        else:
            x,y=N.mgrid[1:10,1:10:0.1]
            self._colorbar_ = self.axcbar.imshow(10-x, cmap=self.scatter_cmap)
        
        if label is not None:
            self.axcbar.set_xlabel(label,fontsize=fontsize_label)
            
        if no_ticks:
            self.axcbar.set_xticks([])
            self.axcbar.set_yticks([])
            
        if add_legend:
            if verticale:
                loc = (0.5,1.)
            else:
                loc = (1.,.5)
            print "legend"
            self.axcbar.text(loc[0],loc[1],r"$\mathrm{%s}$"%self._skyPlot_colored_by_,
                             
                        fontsize="large",
                        va="bottom",ha="center",
                        transform=self.axcbar.transAxes,
                        )
            range_percent_to_show = [0,0.33,0.66,1]
            if verticale:
                [self.axcbar.text(1.02,x,r"$%+.1e$"%(percentile(self._colored_values_,(x*(vmax-vmin) + vmin )*100)),fontsize="small",
                        va="center",ha="left",
                        transform=self.axcbar.transAxes)
                for x in range_percent_to_show]
            else:
                [self.axcbar.text(1.0-x*0.95,-.2,r"$%+.1e$"%(x*100),fontsize="small",
                        va="top",ha="center",
                        transform=self.axcbar.transAxes)
                for x in range_percent_to_show]
                
    #############################
    # -- Internal axis Tools -- #
    #############################
    def _readout_(self,savefile=None,dpi=200):
        """
        """
        if savefile is None:
            self.fig.show()
        else:
            self.fig.savefig(savefile+'.png',dpi=dpi)
            self.fig.savefig(savefile+'.pdf')
            
    def _load_sky_scatter_color_(self,colored_by,default_color="b",
                                 vmin=0,vmax=1):
        """
        """
        self._skyPlot_colored_by_    = colored_by
        self._skyPlot_color_ranges_  = [vmin, vmax]
        if colored_by is None:
            self._colored_values_ = None
            self._color_used_ = default_color
            return None
        
        if colored_by not in dir(self.Samp):
            raise ValueError("Sorry I don't have any %s in module self.Samp"%colored_by)

        self._colored_values_ = N.asarray(self.Samp.__dict__[colored_by])

        self._color_used_     = self.scatter_cmap((self._colored_values_-percentile(self._colored_values_,vmin*100.) )\
                                                   / (percentile(self._colored_values_,vmax*100.)-percentile(self._colored_values_,vmin*100.)))
    
            
        
        
