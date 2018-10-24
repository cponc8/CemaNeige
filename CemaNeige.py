#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 09:04:28 2018

@author: Carine Poncelet
Coded from: Valéry A., Andréassian, V. and Perrin, C.: "As simple as possible but not simpler": What is useful in a temperature-based snow-accounting routine? Part 2 - Sensitivity analysis of the Cemaneige snow accounting routine on 380 catchments, Journal of Hydrology, 517, 1176-1187, doi={10.1016/j.jhydrol.2014.04.058}, 2014.

Verified using: the simulations issued by HOOPLA (https://github.com/AntoineThiboult/HOOPLA). 
Thiboult A., Seiller G., Poncelet C. and Anctil F.: "The HOOPLA toolbox: a HydrOlOgical Prediction LAboratory", submitted to Environmental Modelling & Softwares


INPUTS:
	p   			- time series of total precipitation [mm]
	tair   			- time series of air temperature (°C)
	Date 			- datetime object ('%Y-%m-%d %H:%M:%S')
	QNBV			- average annual snow accumulation [mm]
	AltiBand		- quantiles of elevation [m]
	Z50			- median altitude [m]
	gradT (optional) 	- temperature gradient (°C/100 m)
	Beta (optional)	 	- precipitation correction to elevation (m-1)
	Vmin (optional)		- minimum melting speed
	Tf (optional)		- melting temperature (°C)
	Ctg (optional)		- thermal inertia of the snowpack
	Kf (optional)		- degree-day melting factor (mm/°C)
	

OUTPUTS: 
	Qs :		- sum of liquid precipitation and snowmelt
 
"""


import pandas as pd
import numpy as np
import csv
from numba import jit


class CemaNeige:
    'Enables to run the CemaNeige snow accounting routine'
    
# =============================================================================
# #    Initiate the GR4j object
# =============================================================================
    def __init__(self,BVname,useDefault=1):
        'Provide basic informations on the model - no interaction with the user'
        self.Name = 'CemaNeige'
        self.TimeStep = 'd'
        self.ParName = ['Ctg','Kf']
        self.ParBounds = np.array(([0,0],[1,20])) # minmax param values
        if self.TimeStep == 'd' and useDefault==1:
            self.ParVal = [0.25,3.74]
        
        ## set BV-independent parameters
        self.gradT = [-0.376,-0.374,-0.371,-0.368,-0.366,-0.363,-0.361,-0.358,-0.355,-0.353,-0.350,-0.348,-0.345,-0.343,-0.340,-0.337,-0.335,-0.332,-0.329,-0.327,-0.324,-0.321,-0.319,-0.316,-0.313,-0.311,-0.308,-0.305,-0.303,-0.300,-0.297,-0.295,-0.292,-0.289,-0.287,-0.284,-0.281,-0.279,-0.276,-0.273,-0.271,-0.268,-0.265,-0.263,-0.260,-0.262,-0.264,-0.266,-0.268,-0.270,-0.272,-0.274,-0.277,-0.279,-0.281,-0.283,-0.285,-0.287,-0.289,-0.291,-0.293,-0.295,-0.297,-0.299,-0.301,-0.303,-0.306,-0.308,-0.310,-0.312,-0.314,-0.316,-0.318,-0.320,-0.323,-0.326,-0.330,-0.333,-0.336,-0.339,-0.343,-0.346,-0.349,-0.352,-0.355,-0.359,-0.362,-0.365,-0.368,-0.372,-0.375,-0.378,-0.381,-0.385,-0.388,-0.391,-0.394,-0.397,-0.401,-0.404,-0.407,-0.410,-0.414,-0.417,-0.420,-0.420,-0.421,-0.421,-0.421,-0.422,-0.422,-0.422,-0.423,-0.423,-0.423,-0.424,-0.424,-0.424,-0.425,-0.425,-0.425,-0.426,-0.426,-0.426,-0.427,-0.427,-0.427,-0.428,-0.428,-0.428,-0.429,-0.429,-0.429,-0.430,-0.430,-0.428,-0.425,-0.423,-0.421,-0.419,-0.416,-0.414,-0.412,-0.410,-0.407,-0.405,-0.403,-0.401,-0.398,-0.396,-0.394,-0.392,-0.389,-0.387,-0.385,-0.383,-0.380,-0.378,-0.376,-0.374,-0.371,-0.369,-0.367,-0.365,-0.362,-0.360,-0.362,-0.365,-0.367,-0.369,-0.372,-0.374,-0.376,-0.379,-0.381,-0.383,-0.386,-0.388,-0.390,-0.393,-0.395,-0.397,-0.400,-0.402,-0.404,-0.407,-0.409,-0.411,-0.414,-0.416,-0.418,-0.421,-0.423,-0.425,-0.428,-0.430,-0.431,-0.431,-0.432,-0.433,-0.433,-0.434,-0.435,-0.435,-0.436,-0.436,-0.437,-0.438,-0.438,-0.439,-0.440,-0.440,-0.441,-0.442,-0.442,-0.443,-0.444,-0.444,-0.445,-0.445,-0.446,-0.447,-0.447,-0.448,-0.449,-0.449,-0.450,-0.448,-0.447,-0.445,-0.444,-0.442,-0.440,-0.439,-0.437,-0.435,-0.434,-0.432,-0.431,-0.429,-0.427,-0.426,-0.424,-0.423,-0.421,-0.419,-0.418,-0.416,-0.415,-0.413,-0.411,-0.410,-0.408,-0.406,-0.405,-0.403,-0.402,-0.400,-0.403,-0.405,-0.408,-0.411,-0.413,-0.416,-0.419,-0.421,-0.424,-0.427,-0.429,-0.432,-0.435,-0.437,-0.440,-0.443,-0.445,-0.448,-0.451,-0.453,-0.456,-0.459,-0.461,-0.464,-0.467,-0.469,-0.472,-0.475,-0.477,-0.480,-0.482,-0.483,-0.485,-0.486,-0.488,-0.490,-0.491,-0.493,-0.495,-0.496,-0.498,-0.499,-0.501,-0.503,-0.504,-0.506,-0.507,-0.509,-0.511,-0.512,-0.514,-0.515,-0.517,-0.519,-0.520,-0.522,-0.524,-0.525,-0.527,-0.528,-0.530,-0.526,-0.523,-0.519,-0.515,-0.512,-0.508,-0.504,-0.501,-0.497,-0.493,-0.490,-0.486,-0.482,-0.479,-0.475,-0.471,-0.468,-0.464,-0.460,-0.457,-0.453,-0.449,-0.446,-0.442,-0.438,-0.435,-0.431,-0.427,-0.424,-0.420,-0.417,-0.415,-0.412,-0.410,-0.407,-0.405,-0.402,-0.399,-0.397,-0.394,-0.392,-0.389,-0.386,-0.384,-0.381,-0.379]
        self.Beta = 0
        self.Vmin = 0.1
        self.tf = 0
        
        ## set BV and user dependent parameters
        self.CreateParBV(BVname)
        
    def CreateParBV(self,BVname):
        ifile = './%s_CemaNeigeInfo.csv' % BVname
        csv_file = open(ifile, 'r')
        reader = csv.reader(csv_file)
        CemaInfo = dict(reader)
        self.qnbv = np.float(CemaInfo['QNBV'])
        tmp = CemaInfo['AltiBand']
        tmp = tmp.split(';')
        tmp = [np.float(x) for x in tmp]
        self.zlayers = np.array(tmp)
        self.zmed = np.float(CemaInfo['Z50'])
        self.nbzalt = len(self.zlayers)
        self.c = sum( np.exp( self.Beta*(self.zlayers-self.zmed) ) ) / self.nbzalt
        self.Gthreshold = self.qnbv * 0.9
        
# =============================================================================
# #        Initiate the model
# =============================================================================
    def IniRun(self,inHM,Date,useDefault=1,ParamVal=None):
        'Initiate the model states, unit hydrographs and the outputs of simulation given the parameters and dates suplied by th user'
#        Import parameter values
        if useDefault != 1:
            self.ParVal = np.array(ParamVal)
#        Initiate model states
        self.sta = np.full((2,self.nbzalt),0.)
        
#        Prepare cema inputs
        ileap = np.array(Date.dt.is_leap_year)
        tmp = np.array(Date.dt.dayofyear)
        tmpbool = tmp > 59
        idx = ileap & tmpbool
        tmp[idx] = tmp[idx] - 1
        self.doy = tmp
        
#       Prepare outputs
        Qs = np.empty_like(inHM[0])
        return Qs
        
    
# =============================================================================
#     Run the model
# =============================================================================
    def RunModel(self,inHM,Date,useDefault=1,ParamVal=None):
        'Initiate and run the model'
        Qs = self.IniRun(inHM,Date)
        run_CemaNeige(self.ParVal, inHM[0], inHM[1], Qs, self.sta, self.gradT, self.doy, self.zlayers, self.zmed,self.c, self.nbzalt, self.Beta,self.tf, self.qnbv, self.Vmin,self.Gthreshold )
        return Qs



##--------------------------------------------------------

@jit
def run_CemaNeige( x, p, tair, q , s, gradT, doy, zlayers ,zmed, c, nbzalt, Beta, tf, qnbv, Vmin, Gthreshold ):
    
    for t in range(p.size):
        
# =============================================================================
# FORCING REGIONALIZATION        
# =============================================================================
        theta = gradT[doy[t]-1] # -1 for python indexing
        tz = tair[t] + theta*(zlayers - zmed)/100
        pz = (1/c) * (p[t]/nbzalt) * np.exp( Beta*(zlayers-zmed) )

# =============================================================================
# FRACTION OF SOLID PRECIPITATION
# =============================================================================
       
        Fsol = np.zeros(nbzalt);
        
        for z in range(nbzalt):
            if tz[z] > 3:
                Fsol[z] = 0
            elif tz[z] < -1:
                Fsol[z] = 1
            else:
                Fsol[z] = 1 - ( (tz[z]-(-1)) / ( 3-(-1)) )
        
        pl = (1-Fsol) * pz
        ps = Fsol * pz
        
        
# =============================================================================
# SNOW ACCUMULATION / MELT MODELLING        
# =============================================================================
        
        s[0] +=  ps                          # snowpack content (G)
        tmp = x[0] * s[1] + (1-x[0])*tz
        s[1] = np.minimum(0,tmp)              # thermal state of the snowpack (eTg)
        
        fTg = (s[1] >= tf)                   # melting factor
        fpot = (tz>0) * np.minimum(s[0],x[1]*(tz-tf)*fTg)
        
        fnts = np.minimum( s[0]/Gthreshold,1 )
        
        snowMelt = fpot * ( (1-Vmin) *fnts + Vmin )
        s[0] -= snowMelt
        
        tmp1 = 0.
        tmp2 = 0.
        for z in range(nbzalt):
            tmp1 += pl[z]
            tmp2 += snowMelt[z]
        
        q[t] = tmp1 + tmp2
    







