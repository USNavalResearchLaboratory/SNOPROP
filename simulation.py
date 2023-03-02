# The source code has been authored by federal and non-federal employees. To the extent that a federal employee is an author of a portion of this software or a derivative work thereof, no copyright is claimed by the United States Government, as represented by the Secretary of the Navy ("GOVERNMENT") under Title 17, U.S. Code. All Other Rights Reserved. To the extent that a non-federal employee is an author of a portion of this software or a derivative work thereof, the work was funded in whole or in part by the U.S. Government, and is, therefore, subject to the following license: The Government has unlimited rights to use, modify, reproduce, release, perform, display, or disclose the computer software and computer software documentation in whole or in part, in any manner and for any purpose whatsoever, and to have or authorize others to do so. Any other rights are reserved by the copyright owner.

# Neither the name of NRL or its contributors, nor any entity of the United States Government may be used to endorse or promote products derived from this software, nor does the inclusion of the NRL written and developed software directly or indirectly suggest NRL's or the United States Government's endorsement of this product. 

# THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

import numpy as np
from collections import Iterable
import time
import sys, os
from .FileIO import FileIO
import math
import csv
from scipy import interpolate
#from PMPB import pmpb
from .NumbaMath import calcNeMPI, getRHS
from timeit import default_timer as timer
import scipy.ndimage as nd
from scipy.linalg import eig_banded

complexType = np.complex128
realType = np.float64

class Timer():
    def time(self):
        return timer()
    def __init__(self):
        self.tarr = [self.time()]
        self.sarr = []
    def split(self,name):
        self.sarr.append(name)
        self.tarr.append(self.time())
    def print(self):
        print(self.text())
    def text(self):
        ttot = self.tarr[-1]-self.tarr[0]
        res = ''
        for i in range(len(self.sarr)):
            tsplit = self.tarr[i+1]-self.tarr[i]
            res += '{:.6f} of total ({:.4g}s) spent on {:}'.format(tsplit/ttot,tsplit,self.sarr[i])+'\r\n'
        return res

class Simulation:
    # Constants here. Units are SI.
    c = 299792458. # speed of light in m/s
    e0 = 8.854e-12 # SI vacuum permittivity
    e = 1.602e-19 # Elementary charge in C
    eVtoJ = e # convert eV to J
    me = 9.109e-31 # Electron mass in kg
    hb = 1.0545718e-34 # Reduced Planck's constant
    NH2O = 0 # Users will input neutral density if using ionization

    CFL = 0.8 # Courant-Friedrichs-Lewy scaling factor. Z step size will be this factor times the max dz calculated from the cfl number.
    failed = False
    hankelReady=False
    
    def __init__(self, params):

        # Check that the input deck parameters are valid
        # List the required and optional parameters
        required_params = ['zrange', 'trange', 'tlen', 'rrange', 'rlen',
                           'wV', 'Uion',
                           'sigmaC', 'IMPI', 'eta', 'IBackground', 'include_plasma_refraction',
                           'include_ionization', 'include_energy_loss', 'include_raman',
                           'include_fwm', 'include_kerr', 'include_group_delay', 'include_gvd', 'n2Kerr',
                           'n2Raman', 'wavelength']
        optional_params = [
            'effective_mass', 'material', 'nS', 'nL', 'nA', 'nSg', 'nLg', 'nAg', 'gvd_bS', 'gvd_bL', 'gvd_bA',
            'include_stokes', 'include_antistokes', 'include_collisional_ionization', 'include_radial_derivatives',
            'adaptive_zstep','radial_filter','radial_filter_interval','radial_filter_type','radial_filter_field',
            'profile_L','profile_S','profile_A','dz','dz_min','t_clip',
            'file_output','console_logging_interval','save_restart_interval',
            'save_scalars_interval','save_scalars_z_interval','save_scalars_which',
            'save_1D_interval','save_1D_z_interval','save_1D_which',
            'save_2D_interval','save_2D_z_interval','save_2D_which',
            'lS','lL','lA','N0','cap_Ne','warn_critical','Ne_func',
            'iter_max',
        ]
        possible_params = required_params + optional_params

        # Check whether required parameters exist and that all parameters are acceptable
        missing_params = []
        bad_params = []
        for p in required_params:
            if p not in params:
                missing_params.append(p)
        for p in params:
            if p not in possible_params:
                bad_params.append(p)
        if len(missing_params)>0:
            print('Must specify these parameters for the simulation:')
            print(missing_params)
        if len(bad_params)>0:
            print('Unrecognized parameters in input deck:')
            print(bad_params)
        if len(missing_params)>0 or len(bad_params)>0:
            quit()

        # Retrieve physical constants from input dictionary (all units SI)
        if 'effective_mass' in params: self.me *= params['effective_mass']
        self.include_plasma_refraction, self.include_ionization = params['include_plasma_refraction'], params['include_ionization']
        if self.include_ionization:
            if 'N0' not in params:
                print('Must specify neutral density "N0" if using ionzation')
                quit()
            else:
                self.NH2O = params['N0']
        self.wV, self.Uion, self.sigmaC, self.IMPI = params['wV'], params['Uion']*self.eVtoJ, params['sigmaC'], params['IMPI']
        self.eta, self.IBackground, self.n2Kerr, self.n2Raman = params['eta'], params['IBackground'], params['n2Kerr'], params['n2Raman']
        if 'me' in params: self.me = params['me']
        self.veC = self.NH2O*self.sigmaC * self.e / (np.sqrt(2) * self.me) * 2.0# Electron collision frequency constants (Hafizi 2016 Eq. 9). Adjusted since our A fields are defined as half as large as Bahman's. (E = A*exp() + c.c. here, as opposed to E = 1/2*A*exp() + c.c. in Bahman's paper)
        self.include_collisional_ionization = params['include_collisional_ionization'] if 'include_collisional_ionization' in params else True
        self.viC = self.e**2/(2*self.me*self.Uion) * 4.0 * self.include_collisional_ionization # Avalanche ionization rate constants. Adjusted since our A fields are defined as half as large as Bahman's. (E = A*exp() + c.c. here, as opposed to E = 1/2*A*exp() + c.c. in Bahman's paper)
        
        self.include_energy_loss, self.include_raman = params['include_energy_loss'], params['include_raman']
        #self.include_plasma_refraction *= self.include_ionization
        self.include_energy_loss *= self.include_ionization
        self.include_fwm, self.include_kerr, self.include_group_delay = params['include_fwm'], params['include_kerr'], params['include_group_delay']
        self.include_gvd = params['include_gvd']
        # Include Stokes beam only if SRS is on (or manually turned on/off)
        self.include_stokes = params['include_stokes'] if 'include_stokes' in params else self.include_raman
        # Include Anti-Stokes beam only if SRS is on (or manually turned on/off)
        self.include_antistokes = params['include_antistokes'] if 'include_antistokes' in params else self.include_raman
        self.include_radial_derivatives = params['include_radial_derivatives'] if 'include_radial_derivatives' in params else True
        self.n2Kerr *= self.include_kerr
        self.n2Raman *= self.include_raman
        #self.include_stokes = params['include_stokes'] if 'include_stokes' in params else self.include_raman
        #self.include_antistokes = params['include_antistokes'] if 'include_antistokes' in params else self.include_raman
        self.adaptive_zstep = params['adaptive_zstep'] if 'adaptive_zstep' in params else True
        self.adaptive_zstep_last = 0
        self.radial_filter = params['radial_filter'] if 'radial_filter' in params else False
        self.radial_filter_type = params['radial_filter_type'] if 'radial_filter_type' in params else 'gaussian'
        self.radial_filter_field = params['radial_filter_field'] if 'radial_filter_field' in params else 'electron density'
        self.radial_filter_interval = params['radial_filter_interval'] if 'radial_filter_interval' in params else 1
        self.cap_Ne = params['cap_Ne'] if 'cap_Ne' in params else False
        self.warn_critical = params['warn_critical'] if 'warn_critical' in params else True
        if 'Ne_func' in params and params['Ne_func'] != False:
            if params['include_ionization'] == False:
                self.Ne_func = params['Ne_func']
            else:
                print('Cannot include custom electron density with Ne_func if include_ionization = True')
                quit()
        else: self.Ne_func = False
        self.iterMax = params['iter_max'] if 'iter_max' in params else 12

        # Integration parameters (all units are SI)
        self.profile_L = params['profile_L'] if 'profile_L' in params else False
        self.profile_S = params['profile_S'] if 'profile_S' in params else False
        self.profile_A = params['profile_A'] if 'profile_A' in params else False
        if (self.profile_L == False) and (self.profile_S == False) and (self.profile_A == False):
            print('Must specify a profile for the laser, Stokes, or anti-Stokes beam')
        
        self.material = params['material'] if 'material' in params else 'vacuum'#Segelstein_water'
        self.zstart, self.zend = params['zrange']
        self.tstart, self.tend = params['trange']
        self.rstart, self.rend = params['rrange']
        if self.rstart != 0.:
            print('Simulation must start at r=0! Please check the input for rrange')
            quit()
        self.tlen = params['tlen']
        self.t_clip = params['t_clip'] if 't_clip' in params else 0
        if self.t_clip > (self.tend-self.tstart)/2.0:
            print('"t_clip" parameter must be less than half of the total t range')
            quit()
        self.rlen = params['rlen']
        self.ion_method = params['ionization_method'] if 'ionization_method' in params else 'MPI'
        self.lLv = params['wavelength']
        self.dzMax = params['dz'] if 'dz' in params else 0 # We will calculate the courant condition later if need be
        self.dzMin = params['dz_min'] if 'dz_min' in params else 0 # We will calculate the courant condition later if need be

        # Compute a few more physical constants from the input parameters
        
        self.wL = 2*np.pi*self.c/self.lLv
        self.wS = self.wL-self.wV
        self.wA = self.wL+self.wV
        self.lSv = self.c/self.wS*2*np.pi
        self.lAv = self.c/self.wA*2*np.pi
        self.neCrit = self.e0*self.me/self.e**2*self.wA**2

        # Get refractive index
        if self.material == 'custom':
            # User must specify refractive indices
            required_params = ['nS','nL','nA','nSg','nLg','nAg']#,'gvd_bS','gvd_bL','gvd_bA']
            for p in required_params:
                if p not in params:
                    print('Must specify this parameter to use a custom material: '+p)
                    quit()
            self.nS, self.nL, self.nA = params['nS'],params['nL'],params['nA']
            self.nSg, self.nLg, self.nAg = params['nSg'],params['nLg'],params['nAg']
            self.gvd_bS = params['gvd_bS'] if 'gvd_bS' in params else 0
            self.gvd_bL = params['gvd_bL'] if 'gvd_bL' in params else 0
            self.gvd_bA = params['gvd_bA'] if 'gvd_bA' in params else 0
        else: # IOR will be from a python funcion
            if self.material == 'vacuum':
                ior = lambda e: 1.0
            elif callable(self.material):
                ior = self.material
            #elif os.path.exists(ior_file):
            #    wav, index = [],[]
            #    with open(ior_file) as csvfile:
            #        rd = csv.reader(csvfile, delimiter=',')
            #        done = False
            #        for row in rd:
            #            if row[0] is not '':
            #                wav.append(float(row[0]))
            #                index.append(float(row[1]))
            #    ior = interpolate.interp1d(np.array(wav)*1e-6, index,kind='quadratic')
            else:
                print("'material' parameter must be one of 'custom', 'vacuum', or a python function")
                quit()
                
            self.nL = ior(self.lLv)
            self.nS = ior(self.lSv)
            self.nA = ior(self.lAv)

            # get group refractive index
            # get derivative of n0 at laser wavelength by taking difference between n0 at lLv+dlLv and lLv-dlLv; dlLv is just the sampling distance on either side of the wavelength of interest
            def getNg(l0,dlLv=1e-9):
                dn0dl = (ior(l0+dlLv) - ior(l0-dlLv)) / (2*dlLv)
                return ior(l0) - l0*dn0dl
            self.nSg = getNg(self.lSv) # Group index
            self.nLg = getNg(self.lLv)
            self.nAg = getNg(self.lAv)


            # get GVD parameters
            def getGVDBeta(l1,domega=self.wL*0.01):
                omega1 = self.c/l1*2*np.pi
                omega0, omega2 = omega1-domega, omega1+domega
                l0, l2 = self.c/(omega0/2/np.pi), self.c/(omega2/2/np.pi)
                k0, k1, k2 = omega0*ior(l0)/self.c, omega1*ior(l1)/self.c, omega2*ior(l2)/self.c
                if k2-k0 != 0:
                    d2kdw2 = (k0+k2-2*k1)/domega**2
                else:
                    d2kdw2 = 0
                return d2kdw2
            self.gvd_bS = getGVDBeta(self.lSv)
            self.gvd_bL = getGVDBeta(self.lLv)
            self.gvd_bA = getGVDBeta(self.lAv)

        if not self.include_gvd:
            self.gvd_bS, self.gvd_bL, self.gvd_bA = 0,0,0

        self.vSg = self.c / self.nSg # Group velocity
        self.vLg = self.c / self.nLg
        self.vAg = self.c / self.nAg
        
        self.kS = 2*np.pi*self.nS/self.lSv
        self.kL = 2*np.pi*self.nL/self.lLv
        self.kA = 2*np.pi*self.nA/self.lAv
        self.antistokesAngle = np.arccos((self.kA**2+4*self.kL**2-self.kS**2)/(4*self.kA*self.kL)) # magic angle for Anti-Stokes propagation (amplified four-wave mixing)
        self.stokesAngle = np.arccos((self.kS**2+4*self.kL**2-self.kA**2)/(4*self.kS*self.kL)) # magic angle for Stokes propagation (which leads to amplified four-wave mixing)
        self.dk = 2*self.kL-self.kA-self.kS


        

        
        if 'lS' in params: self.lS = params['lS']
        else: self.lS = int(self.Uion / (self.hb*self.wS)) + 1 # How many photons to ionize water
        if 'lL' in params: self.lL = params['lL']
        else: self.lL = int(self.Uion / (self.hb*self.wL)) + 1 # How many photons to ionize water
        if 'lA' in params: self.lA = params['lA']
        else: self.lA = int(self.Uion / (self.hb*self.wA)) + 1 # How many photons to ionize water
        self.EBgS = self.getEField(self.IBackground,self.nS) # Get background E-field from background intensity
        self.EBgL = self.getEField(self.IBackground,self.nL) # Get background E-field from background intensity
        self.EBgA = self.getEField(self.IBackground,self.nA) # Get background E-field from background intensity
        self.xNR = self.n2Kerr * 4*self.nL**2*self.e0*self.c/3 # convert n2 at laser wavelength to xNR using Boyd Eq. 4.1.19 (not 1.2.14b)
        self.xRS = self.n2Raman * 4*self.nL**2*self.e0*self.c/3 # convert n2 at laser wavelength to xNR using Boyd Eq. 4.1.19 (not 1.2.14b)
        self.xRA = -1.*self.xRS

        self.errorThreshold = 0.01 # Threshold for crank-nicolson iteration

        # Basic output parameters
        self.saveRestartInterval = params['save_restart_interval'] if 'save_restart_interval' in params else 0 # Interval in frames for full saves
        self.save2DInterval = params['save_2D_interval'] if 'save_2D_interval' in params else 0 # Interval in frames for 2D saves
        self.save2DZInterval = params['save_2D_z_interval'] if 'save_2D_z_interval' in params else 0 # Interval in frames for 2D saves
        self.save2DWhich = params['save_2D_which'] if 'save_2D_which' in params else [] # Interval in frames for 2D saves
        self.saveScalarsInterval = params['save_scalars_interval'] if 'save_scalars_interval' in params else 0 # Interval in frames for scalars
        self.saveScalarsZInterval = params['save_scalars_z_interval'] if 'save_scalars_z_interval' in params else 0 # Interval in frames for scalars
        if 'save_scalars_which' in params and params['save_scalars_which'] != []:
            self.saveScalarsWhich = ['step','z'] + params['save_scalars_which']
        else: self.saveScalarsWhich = []
        self.save1DInterval = params['save_1D_interval'] if 'save_1D_interval' in params else 0 # Interval in frames for 1D saves
        self.save1DZInterval = params['save_1D_z_interval'] if 'save_1D_z_interval' in params else 0 # Interval in frames for 1D saves
        self.save1DWhich = params['save_1D_which'] if 'save_1D_which' in params else [] # Interval in frames for 1D saves

        # Switches for managing output
        self.saveRestarts = self.saveRestartInterval > 0
        self.save2D = ((self.save2DInterval > 0) or self.save2DZInterval > 0) and len(self.save2DWhich) != 0
        self.saveScalars = ((self.saveScalarsInterval > 0) or (self.saveScalarsZInterval > 0)) and len(self.saveScalarsWhich) != 0
        self.save1D = ((self.save1DInterval > 0) or (self.save1DZInterval > 0))
        self.file_output = params['file_output'] if 'file_output' in params else True # Set this to false to avoid writing any output files
        if self.file_output == False:
            self.save2D, self.saveScalars, self.saveRestarts, self.save1D = False, False, False, False # Cut all output if params['file_output'] is false
        self.console_logging_interval = params['console_logging_interval'] if 'console_logging_interval' in params else 20 # Set this to 0 to eliminate console logging
        
        self.logText = []


        if self.file_output:
            self.filer = FileIO() # Makes folder to store the data
            if self.saveScalars: # Do this now, after self.file_output has been checked, to initialize the scalar save file
                self.scalarsFile = self.filer.makeCSVFile('scalars', self.saveScalarsWhich)
        else:
            self.filer = False
            
        self.mesh_init()
        #if 'force_FWHM' in params:
        #    scaleFrac = params['force_FWHM']/self.getFWHM(self.getFluence(self.getIL()+self.getIA()+self.getIS()))
        #    print(scaleFrac)
        #    self.Rpulse *= scaleFrac
        #    self.mesh_init()
        

    def mesh_init(self): # Create the simulation data structures and initialize the pulses
        self.dt = (self.tend-self.tstart)/(self.tlen) if self.tlen > 1 else (self.tend-self.tstart)
        self.dr = (self.rend-self.rstart)/(self.rlen-1)
        # Courant condition. Get the z step size.
        #utau = (self.nLg-self.nSg)/self.c * self.include_group_delay # no need to check this if we aren't calculating the Raman beam or shift
        #ur = 1.j*self.c/(2*self.wL*self.nL)
        #if self.dzMax == 0: # Need to calculate dzMax from the Courant condition
        #    self.dzMax = float(np.abs(self.CFL/(utau/self.dt+ur/self.dr**2)))
        utau = (1/self.nLg-1/self.nAg) * self.include_group_delay # no need to check this if we aren't calculating the Raman beam or shift
        ur = self.c * .12 # Beam propagating with sin(angle)=0.1 off axis
        if self.dzMax == 0: # Need to calculate dzMax from the Courant condition
            self.dzMax = self.CFL * self.c/(ur/self.dr + utau/self.dt)
        #print('CFL')
        #print('cfl t',1/(utau/self.dt),self.c/((1/self.nLg-1/self.nAg)/self.dt))
        #print('cfl r',1/(ur/self.dr**2),self.c/(self.c/self.dr * .1))
        #print('dz total',self.dzMax,self.c/(self.c/self.dr * .1 + (1/self.nLg-1/self.nAg)/self.dt))
        #quit()
        self.dz = self.dzMax
        if self.dzMin==0: self.dzMin = self.dzMax / 10**4
        self.zlen = int(np.ceil((self.zend-self.zstart)/self.dz)) + 1

        # Initialize the grid
        self.r0 = np.linspace(self.rstart, self.rend-self.dr, self.rlen)#+self.dr/2.
        self.t0 = np.linspace(self.tstart, self.tend, self.tlen)
        self.rr, self.tt = np.meshgrid(self.r0,self.t0) # Indexing will be A(t,r)
        self.ve = np.zeros((self.tlen,self.rlen), dtype=realType)
        self.ne = np.zeros((self.tlen,self.rlen), dtype=realType)
        self.te = np.zeros((self.tlen,self.rlen), dtype=realType)              # comment GMP: added array te (set all elements to 0)
        self.te += 0.03                                                        # comment GMP: initialized array te (room temperature)
        if self.Ne_func != False:
            self.ne += self.Ne_func(self.tt,self.rr)
        self.WMPI_NH2O   = np.zeros((self.tlen,self.rlen), dtype=realType)
        self.WMPI_NH2O_S = np.zeros((self.tlen,self.rlen), dtype=realType)
        self.WMPI_NH2O_L = np.zeros((self.tlen,self.rlen), dtype=realType)
        self.WMPI_NH2O_A = np.zeros((self.tlen,self.rlen), dtype=realType)
        self.plasmaEnergy = 0.
        self.rrinv = np.zeros((self.tlen,self.rlen))
        self.rrinv[:,1:] = self.rr[:,1:]**-1
        self.rrinv[:,0] = 0 # Otherwise it is infinite. This is ok since the derivative must be zero at r=0 anyways.
        self.AL = np.zeros((self.tlen,self.rlen), dtype=complexType)
        self.AS = np.zeros((self.tlen,self.rlen), dtype=complexType)
        self.AA = np.zeros((self.tlen,self.rlen), dtype=complexType)


        # Set up radial filter if need be
        if self.radial_filter and self.radial_filter_type == 'hankel':
            self.setupHankel()

        # Initialize the fields. Since I scales as exp(-2(t/sigmaT)^2), E scales as exp(-(t/sigmaT)^2)
        # Assume intensity scales as I0*exp(-2(t/sigmaT)^2)*exp(-2(r/sigmaR)^2)
        fields_to_setup = []
        if self.profile_L != False:
            fields_to_setup.append([self.AL,self.lLv,self.nL,self.profile_L,'profile_L']) # Optional Stokes field input
        if self.profile_S != False:
            fields_to_setup.append([self.AS,self.lSv,self.nS,self.profile_S,'profile_S']) # Optional Stokes field input
        if self.profile_A != False:
            fields_to_setup.append([self.AA,self.lAv,self.nA,self.profile_A,'profile_A']) # Optional Anti-Stokes field input

        for field,wavelength,n,params,name in fields_to_setup:
            if '2D_data' in params:
                if params['2D_data'].shape[0] == self.tlen and params['2D_data'].shape[1] == self.rlen:
                    field[:,:] = params['2D_data']
                    if 'energy' in params:
                        field[:,:] *= np.sqrt(params['energy']/self.getEnField(field,n)) # Now scale the whole multi-pulse E-field profile to get the right energy.
                    if 'IBackground' in params:
                        Amin = self.getA(self.getEField(params['IBackground'],n))
                        #field[np.abs(field)<Amin] = Amin
                        field[:,:] += Amin
                        
                else:
                    print('"2D_data" element must have shape (tlen, rlen)! Currently has shape ',params['2D_data'].shape)
            else:
                params_needed = ['pulse_length_fwhm','toffset','efrac']
                for p in params_needed:
                    if p not in params:
                        print('Must either specify "2D_data" or specify parameter "'+p+'" in '+name+' dictionary')
                        quit()
                if 'energy' not in params and 'intensity' not in params:
                    print('Must specify either "energy" or "intensity" in '+name+' dictionary')
                    quit()
                k = 2*np.pi*n/wavelength

                tau = np.array(params['pulse_length_fwhm'])
                temporalData = params['temporal_data'] if 'temporal_data' in params else []
                radialData = params['radial_data'] if 'radial_data' in params else []
                temporalFunc = params['temporal_func'] if 'temporal_func' in params else False
                radialFunc = params['radial_func'] if 'radial_func' in params else False
                if len(temporalData)>0 and temporalFunc != False:
                    print("Cannot specify both temporal_data and temporal_func for any pulse")
                    quit()
                if len(radialData)>0 and radialFunc != False:
                    print("Cannot specify both radial_data and radial_func for any pulse")
                    quit()
                toffset = np.array(params['toffset'])
                efrac = np.array(params['efrac'])
                efrac = efrac / np.sum(efrac) # Normalize it if it wasn't already
                if 'pulse_radius_e2' in params:
                    Rpulse = np.array(params['pulse_radius_e2'])
                    sigmaR = Rpulse
                elif 'pulse_radius_half' in params:
                    fwhm = np.array(params['pulse_radius_half'])
                    Rpulse = fwhm*np.sqrt(2./np.log(2.)) # Get 1/e^2 radius
                    sigmaR = Rpulse
                else:
                    if (len(radialData) == 0) and (radialFunc == False):
                        print('Must specify radial profile for the pulse '+name)
                        quit()

                if len(temporalData)==0 and temporalFunc == False:
                    sigmaT = tau/2*np.sqrt(2/np.log(2)) # 1/e^2 half-length


                # Start with the laser profile
                pumpPulsesI = []
                # Set up the different pulses separately before combining them
                for i in range(len(tau)):
                    if len(temporalData) != 0:
                        temporalFunc2 = interpolate.interp1d(temporalData[0],temporalData[1])
                        pumpPulseI = temporalFunc2(self.tt)
                    elif temporalFunc != False:
                        pumpPulseI = temporalFunc(self.tt)
                    elif self.tlen==1:
                        pumpPulseI = np.ones(self.tt.shape)
                    else:
                        pumpPulseI = np.exp(-2*np.square((self.tt-toffset[i])/sigmaT[i]))
                    if self.t_clip != 0: # Clip near edges to avoid boundary issues
                        tmid = (self.tend+self.tstart)/2.0
                        width_no_clip = (self.tend-self.tstart)/2.0 - self.t_clip
                        pumpPulseI[(np.abs(self.tt - tmid) > width_no_clip)] = 0.0
                    if len(radialData) != 0:
                        rad_r, rad_d = radialData
                        if rad_r[-1] < self.rend: # Make it go to zero at large r if necessary
                            dr = rad_r[-1]-rad_r[-2]
                            rad_r=np.append(rad_r,rad_r[-1] + dr)
                            rad_r=np.append(rad_r,self.rend+dr)
                            rad_d=np.append(rad_d,0.)
                            rad_d=np.append(rad_d,0.)
                        radialFunc2 = interpolate.interp1d(rad_r,rad_d)
                        pumpPulseI *= radialFunc2(self.rr)
                    elif radialFunc != False:
                        pumpPulseI *= radialFunc(self.rr)
                    else:
                        pumpPulseI *= np.exp(-2*np.square(self.rr/sigmaR[i])) # radial profile 

                    pumpPulsesI.append(pumpPulseI)

                    pumpPulsesI[i] *= efrac[i]/efrac[0]* self.getEn(pumpPulsesI[0])/self.getEn(pumpPulsesI[i]) # scale pulse energy relative to first
                pumpPulsesI = np.array(pumpPulsesI)
                pumpProfileI = np.sum(pumpPulsesI, axis=0)
                if (pumpProfileI<0).any():
                    print('Final pump energy must not be negative!')
                    quit()
                if 'energy' in params:
                    pumpProfileI *= params['energy']/self.getEn(pumpProfileI) # Now scale the whole multi-pulse E-field profile to get the right energy.
                else: # then intensity must be specified
                    pumpProfileI *= params['intensity']/np.max(pumpProfileI)
                pumpProfileE = self.getEField(pumpProfileI,n).astype(complexType)

                if 'axicon_angle' in params:
                    angle = params['axicon_angle']*-1
                    if angle == 0: angle = 1e-10                    
                    pumpProfileE *= np.exp(1.j*k*self.rr*np.sin(angle)) # wavefront curvature (initial focusing) for axicon lens
                elif 'focal_length' in params:
                    f = params['focal_length']
                    #focusAngle = np.arctan2(sigmaR[0], f)
                    #w0L = wavelength/n/np.pi/focusAngle
                    #zRL = np.pi*w0L**2/(wavelength/n)
                    Rz = -1*f#(f+zRL**2/f) # Radius of curvature to get the desired focal length.
                    pumpProfileE *= np.exp(1.j*k*self.rr**2/2/Rz) # wavefront curvature (initial focusing)
                else:
                    #if ('focal_length' in params) + ('axicon_angle' in params) != 1:
                    print('Must specify one of "focal_length" or "axicon_angle" in '+name)
                    quit()

                # Add the pulse profile to the field
                field[:,:] += self.getA(pumpProfileE) # AL is the envelope field, defined with E=A exp(iwt) + c.c.

                # Add background field
                IBackground = params['IBackground'] if 'IBackground' in params else self.IBackground 
                Amin = self.getA(self.getEField(IBackground,n))
                #field[np.abs(field)<Amin] = Amin
                field[:,:] += Amin
               
            
        # Add backgrounds to these fields if they didn't have specified profiles
        # Also set update flags
        if self.profile_S == False:
            ASmin = self.getA(self.getEField(self.IBackground,self.nS))
            #self.AS[np.abs(self.AS)<ASmin] = ASmin
            self.AS += ASmin
            self.updateS=True
        else:
            self.updateS = params['update'] if 'update' in params else True
        if self.profile_L == False:
            ALmin = self.getA(self.getEField(self.IBackground,self.nL))
            #self.AS[np.abs(self.AS)<ASmin] = ASmin
            self.AL += ALmin
            self.updateL=True
        else:
            self.updateL = params['update'] if 'update' in params else True
        if self.profile_A == False:
            AAmin = self.getA(self.getEField(self.IBackground,self.nA))
            #self.AA[np.abs(self.AA)<AAmin] = AAmin
            self.AA += AAmin
            self.updateA=True
        else:
            self.updateA = params['update'] if 'update' in params else True

        self.energy = self.getEnS()+self.getEnL()+self.getEnA()
            
        self.z = self.zstart

        self.timerStart = time.time()*1000.0
        self.stepTime = time.time()
        self.meanStepTime = 0
        self.CNcount = 0.
        self.step = 0



    def setupHankel(self):
        self.hankelReady=True
        rcenter = self.r0+self.dr/2.
        V = np.pi*((rcenter+0.5*self.dr)**2 - (rcenter-0.5*self.dr)**2)
        A1 = 2*np.pi*(rcenter-0.5*self.dr)
        A2 = 2*np.pi*(rcenter+0.5*self.dr)
        self.filter_lambda = np.sqrt(V)
        T1 = A1/(self.dr*V)
        T2 = -(A1 + A2)/(self.dr*V)
        T3 = A2/(self.dr*V)
        # Boundary conditions
        T2[0] += T1[0]
        T2[-1] -= T3[-1]
        # Symmetrize the matrix S = Lambda * T * Lambda^-1
        # This is the root-volume weighting
        T1[1:] *= self.filter_lambda[1:]/self.filter_lambda[:-1]
        T3[:-1] *= self.filter_lambda[:-1]/self.filter_lambda[1:]
        a_band_upper = np.zeros((2,self.rlen))
        a_band_upper[0,:] = T1 # T3->T1 thanks to scipy packing and symmetry
        a_band_upper[1,:] = T2
        vals, self.Hi = eig_banded(a_band_upper) # Columns of Hi are the radial modes
        vals, self.Hi = np.flip(vals), np.flip(self.Hi,axis=1) # Make the first column the lowest frequency
        # Save the filter k space profile as well. We want to filter out about the top half of modes.
        #self.radialFilterShape = np.ones(len(self.r0))
        #self.radialFilterShape[int(self.rlen/2):] = 0
        #i1, i2 = int(self.rlen*.25), int(self.rlen*.75)
        #r1, r2 = self.r0[i1],self.r0[i2]
        #self.radialFilterShape[i1:i2] = np.cos(np.pi*(self.r0[i1:i2]-r1)/(r2-r1))*.5 + .5
        self.radialFilterShape = np.cos(np.pi*(np.cos(np.pi*(self.r0/self.r0[-1]))*.5+.5))*-.5+.5

            
        

    # Constants to be simplify the integration
    def constants_gen(self): 
        if self.include_raman is False:
            self.xRA, self.xRS = 0.,0.
        if self.include_kerr is False:
            self.xNR = 0.

        self.cS1 = 1j*self.c/(2*self.nS*self.wS) * self.include_radial_derivatives
        self.cS2 = (self.nLg-self.nSg)/self.c * self.include_group_delay
        self.cS3 = 3j*self.wS/(self.nS*self.c)*.5*self.xNR
        self.cS4 = 3j*self.wS/(self.nS*self.c)*(self.xRS+self.xNR)
        self.cS5 = 3j*self.wS/(self.nS*self.c)*self.xNR
        self.cS6 = self.cS4
        self.cS7 = 1.0/(2j*self.kS)* (self.e**2/(self.me*self.e0))/self.c**2 * self.include_plasma_refraction
        self.cS8 = self.cS7*(-1.j/self.wS) * self.include_energy_loss
        self.cS9 = -self.wS/self.kS/(self.c*self.c)*self.Uion/4./self.e0 * self.include_energy_loss
        self.cS10 = self.gvd_bS/2j * self.include_gvd

        self.cL1 = 1j*self.c/(2*self.nL*self.wL) * self.include_radial_derivatives
        self.cL2 = 0.
        self.cL3 = 3j*self.wL/(self.nL*self.c)*(self.xRA+self.xNR)
        self.cL4 = 3j*self.wL/(self.nL*self.c)*.5*self.xNR
        self.cL5 = 3j*self.wL/(self.nL*self.c)*(self.xRS+self.xNR)
        self.cL6 = 3j*self.wL/(self.nL*self.c)*2*self.xNR
        self.cL7 = 1.0/(2j*self.kL)* (self.e**2/(self.me*self.e0))/self.c**2 * self.include_plasma_refraction
        self.cL8 = self.cL7*(-1.j/self.wL) * self.include_energy_loss
        self.cL9 = -self.wL/self.kL/(self.c*self.c)*self.Uion/4./self.e0 * self.include_energy_loss
        self.cL10 = self.gvd_bL/2j * self.include_gvd

        self.cA1 = 1j*self.c/(2*self.nA*self.wA) * self.include_radial_derivatives
        self.cA2 = (self.nLg-self.nAg)/self.c * self.include_group_delay
        self.cA3 = 3j*self.wA/(self.nA*self.c)*self.xNR
        self.cA4 = 3j*self.wA/(self.nA*self.c)*(self.xRA+self.xNR)
        self.cA5 = 3j*self.wA/(self.nA*self.c)*.5*self.xNR
        self.cA6 = self.cA4
        self.cA7 = 1.0/(2j*self.kA)* (self.e**2/(self.me*self.e0))/self.c**2 * self.include_plasma_refraction
        self.cA8 = self.cA7*(-1.j/self.wA) * self.include_energy_loss
        self.cA9 = -self.wA/self.kA/(self.c*self.c)*self.Uion/4./self.e0 * self.include_energy_loss
        self.cA10 = self.gvd_bA/2j * self.include_gvd



        # # Check the constants again
        # print('cS')
        # print(self.cS1, -1/(2j*self.kS))


        #print(self.cS2, (2j*(self.nS**2*self.wS/self.c**2 - self.kS*self.nL/self.c)/(2j*self.kS)))
        #print(self.cS2, (2j*(self.nS**2*self.wS/self.c**2 - self.kS/self.vg)/(2j*self.kS)))
        #print(self.cA2, (2j*(self.nA**2*self.wA/self.c**2 - self.kA*self.nL/self.c)/(2j*self.kA)))
        #print(self.cA2, (2j*(self.nA**2*self.wA/self.c**2 - self.kA/self.vg)/(2j*self.kA)))
        #print(self.c/self.nS, self.c/self.nL, self.c/self.nA, self.vg)


        # print(self.cS3, -6*self.wS**2/self.c**2*(1/2*self.xNR)/(2j*self.kS))
        # print(self.cS4, -6*self.wS**2/self.c**2*(self.xRS+self.xNR)/(2j*self.kS))
        # print(self.cS5, -6*self.wS**2/self.c**2*(self.xNR)/(2j*self.kS))
        # print(self.cS6, -6*self.wS**2/self.c**2*(self.xRS+self.xNR)/(2j*self.kS))
        # print(self.cS7, self.e**2/(self.me*self.e0*self.c**2)/(2j*self.kS))
        # print(self.cS8, self.e**2/(self.me*self.e0*self.c**2)*(-1j/self.wS)/(2j*self.kS))
        #print(self.cS9, -1j*self.wS/(2*self.c**2)*self.Uion/(2j*self.kS)/self.e0)

        # print('cL')
        # print(self.cL1, -1/(2j*self.kL))
        # print(self.cL2, 0/(2j*self.kL))
        # print(self.cL3, -6*self.wL**2/self.c**2*(self.xRA+self.xNR)/(2j*self.kL))
        # print(self.cL4, -6*self.wL**2/self.c**2*(1/2*self.xNR)/(2j*self.kL))
        # print(self.cL5, -6*self.wL**2/self.c**2*(self.xRS+self.xNR)/(2j*self.kL))
        # print(self.cL6, -6*self.wL**2/self.c**2*(self.xNR*2)/(2j*self.kL))
        # print(self.cL7, self.e**2/(self.me*self.e0*self.c**2)/(2j*self.kL))
        # print(self.cL8, self.e**2/(self.me*self.e0*self.c**2)*(-1j/self.wL)/(2j*self.kL))
        #print(self.cL9, -1j*self.wL/(2*self.c**2)*self.Uion/(2j*self.kL)/self.e0)

        # print('cA')
        # print(self.cA1, -1/(2j*self.kA))
        # print(self.cA2, 0/(2j*self.kA))
        # print(self.cA3, -6*self.wA**2/self.c**2*(self.xNR)/(2j*self.kA))
        # print(self.cA4, -6*self.wA**2/self.c**2*(self.xRA+self.xNR)/(2j*self.kA))
        # print(self.cA5, -6*self.wA**2/self.c**2*(1/2*self.xNR)/(2j*self.kA))
        # print(self.cA6, -6*self.wA**2/self.c**2*(self.xRA+self.xNR)/(2j*self.kA))
        # print(self.cA7, self.e**2/(self.me*self.e0*self.c**2)/(2j*self.kA))
        # print(self.cA8, self.e**2/(self.me*self.e0*self.c**2)*(-1j/self.wA)/(2j*self.kA))
        #print(self.cA9, -1j*self.wA/(2*self.c**2)*self.Uion/(2j*self.kA)/self.e0)
        #quit()

        # Put the constants into a single variable to pass to our Numba functions
        self.constants = (
            (self.cS1, self.cS2, self.cS3, self.cS4, self.cS5, self.cS6, self.cS7, self.cS8, self.cS9, self.cS10),
            (self.cL1, self.cL2, self.cL3, self.cL4, self.cL5, self.cL6, self.cL7, self.cL8, self.cL9, self.cL10),
            (self.cA1, self.cA2, self.cA3, self.cA4, self.cA5, self.cA6, self.cA7, self.cA8, self.cA9, self.cA10),
            (self.dt, self.dr, self.tlen, self.rlen)
            )
        self.includes = (self.include_stokes, self.include_antistokes, self.include_ionization, self.include_plasma_refraction, self.include_energy_loss,self.updateS,self.updateL,self.updateA)


    def radialFilter(self, field):
        if not self.radial_filter:
            print('Must turn on radial filter before initialization to use radial filter.')
            quit()
        if self.radial_filter_type == 'gaussian':
            return nd.gaussian_filter1d(field,1)
        elif self.radial_filter_type == 'hankel':
            output = self.radialHankel(field) # Only the r-direction is in kspace. But that's OK since our filter is t-independent.
            output = output*self.radialFilterShape # Throw out the high modes
            return self.invRadialHankel(output)
        else:
            print("Only Gaussian (real-space) and Hankel (k-space) filters are permitted.")
            print("Please set radial_filter_type to 'gaussian' or 'hankel'")
    def radialHankel(self, v):
        if not self.hankelReady:
            self.setupHankel()
        return self.radialScaleAndMultiply(v,self.Hi,self.filter_lambda)
    def invRadialHankel(self, v):
        if not self.hankelReady:
            self.setupHankel()
        return self.radialScaleAndMultiply(v,self.Hi.T,self.filter_lambda)
    def radialScaleAndMultiply(self,v, mat, scale):
        v = np.einsum('j,ij...->ij...',scale,v)
        v = np.einsum('ij,fi->fj',mat,v) # Do the filter
        v = np.einsum('j,ij...->ij...',1/scale,v)
        return v

    # Helpers for useful metrics and common conversions
    def getInten(self, A, n): # Get intensity from field where A is the envelope field (E= A exp(i w t) + c.c.)
        return 4.*np.absolute(A*np.conj(A))*(n/2.*self.e0*self.c) # See e.g. Boyd 2.5.1
    def getFluence(self,inten):
        return np.sum(inten,axis=0)*self.dt
    def getEField(self, I, n): return np.sqrt(I/(n/2*self.e0*self.c))
    def getAFromI(self, I, n): return np.sqrt(I/(n*2*self.e0*self.c))



    # Public methods
    def getZ(self): return self.z
    def getBeamParams(self): return {
            'lambdaS': self.lSv,'lambdaL': self.lLv,'lambdaA': self.lAv,
            'nS': self.nS, 'nL': self.nL, 'nA': self.nA,
            'nSg': self.nS, 'nLg': self.nL, 'nAg': self.nA,
            'beta_gS': self.nS, 'beta_gL': self.nL, 'beta_gA': self.nA,
    }
    def getGrid(self): return {
            'trange': [self.tstart,self.tend],'tlen':self.tlen,
            'rrange': [self.rstart,self.rend],'rlen':self.rlen,
            'zrange': [self.zstart,self.zend],'dz':self.dz, 'z': self.z,
            'r':self.r0, 't': self.t0, 'rr': self.rr, 'tt': self.tt,
    }

    def getEn(self, inten): return np.sum(inten*(self.rr)*self.dr*self.dt*2*np.pi)
    def getEnField(self, A, n): # Get energy of an field where A is the envelope field (E= A exp(i w t) + c.c.)
        return np.sum(self.getInten(A,n)*self.rr*self.dr*self.dt*2*np.pi)
    def getEnS(self): return self.getEn(self.getIS())
    def getEnL(self): return self.getEn(self.getIL())
    def getEnA(self): return self.getEn(self.getIA())
    def getIS(self): return self.getInten(self.AS,self.nS)
    def getIL(self): return self.getInten(self.AL,self.nL)
    def getIA(self): return self.getInten(self.AA,self.nA)
    def getAS(self): return self.AS
    def getAL(self): return self.AL
    def getAA(self): return self.AA
    def getES(self): return self.getEFromA(self.AS)
    def getEL(self): return self.getEFromA(self.AL)
    def getEA(self): return self.getEFromA(self.AA)
    def setAS(self,A): self.AS = A.copy()
    def setAL(self,A): self.AL = A.copy()
    def setAA(self,A): self.AA = A.copy()


    def getP(self,I): return 2*np.pi*np.sum(I*self.rr,axis=1)*self.dr
    def getA(self, E): return E/2. # Using definition E= A exp(i w t) + c.c.
    def getEFromA(self, A): return 2.*A
    def getFWHM(self, fluence):
        return 2.*self.getRadiusFraction(0.5,fluence)
    def getRMSSize(self, fluence):
        return 2*np.sqrt(np.sum(fluence*self.r0**2)/np.sum(fluence))
    def getRadiusFraction(self,fraction,fluence): # Get the radius to a fraction of max fluence
        fluenceFrac = fraction*np.max(fluence)
        overFrac = fluence>fluenceFrac
        if sum(overFrac) == 0: return 0.
        idxHalf = np.max(np.where(overFrac)[0])+1
        if idxHalf >= self.rlen: return 0.
        fBelow = fluence[idxHalf]
        fAbove = fluence[idxHalf-1]
        fAboveDiff = fAbove-fluenceFrac
        fBelowDiff = fluenceFrac-fBelow
        fractionAlong = fAboveDiff/(fAboveDiff+fBelowDiff)
        return self.r0[idxHalf-1] + self.dr*fractionAlong

        
    def display(self, number):
        if (type(number) is str or isinstance(number, Iterable)):
            return str(number)
        else:
            return "{:.6g}".format(number)
    def printIfConsoleOutput(self, *string):
        if self.console_logging_interval != 0:
            if len(string)==1:
                print(string[0])
            else:
                print(string)
    def log(self, message):
        self.logText.append(message)
    def logSave(self,name):
        if self.file_output:
            self.filer.saveTextFile(self.logText,name)
    def logInit(self):
        self.logText = []
        self.log('----Constants----All units are SI unless otherwise stated----')
        textPairs = [('Uion',self.Uion),('Uion (eV)',self.Uion/self.e),('NH2O',self.NH2O),
                    ('IMPI',self.IMPI),('collision cross section',self.sigmaC),
                    ('Electron attachment rate',self.eta),('xNR',self.xNR),('xRS',self.xRS),('xRA',self.xRA),
                    ('Background intensity',self.IBackground),('Background field (laser)',self.EBgL),
                    ('Background field (Stokes)',self.EBgS),('Background field (Anti-Stokes)',self.EBgA)]
        for pair in textPairs: self.log(pair[0]+': '+self.display(pair[1]))
        self.log('----Pulse Parameters----')
        textPairs = [('Stokes vacuum wavelength',self.lSv),('Laser vacuum wavelength',self.lLv),('Anti-Stokes vacuum Wavelength',self.lAv),
                    ('nS',self.nS),('nL',self.nL),('nA',self.nA),
                    ('nSg',self.nSg),('nLg',self.nLg),('nAg',self.nAg),('dk',self.dk),
                    ('gvd_bS',self.gvd_bS),('gvd_bL',self.gvd_bL),('gvd_bA',self.gvd_bA),
                    #('F-number (using 1/e^2 diameter)', self.f/(2*self.Rpulse)),('Focal length',self.f),
                    ('Pulse energy',self.energy),#('Energy fractions',self.efrac),
                    ('Max power',np.max(np.sum(self.getIL()*2*np.pi*self.r0*self.dr,1))),
                    #('Pulse length (FWHM)',self.tau),('Spot radius (1/e^2)',self.Rpulse),
                    ('Peak laser intensity at start',np.max(self.getIL())),('Peak laser field at start',np.max(np.absolute(self.AL))),
                    ('Stokes starting field',np.max(self.AS)),('Anti-Stokes starting field',np.max(self.AA))]
        for pair in textPairs: self.log(pair[0]+': '+self.display(pair[1]))
        self.log("optimal Stokes angle (degrees):"+self.display(self.stokesAngle*180./np.pi))
        self.log("optimal anti-Stokes angle (degrees):"+self.display(self.antistokesAngle*180./np.pi))
        self.log("dk over kL:"+self.display(self.dk/(2*np.pi*self.nL/self.lLv)))
        #self.log("dk*radius:"+self.display(self.dk*self.Rpulse))
        self.log('----Numerical Parameters----')
        textPairs = [('Ionization method',self.ion_method),('MPI photons to ionize (l Stokes)',self.lS),
                     ('MPI photons to ionize (l Laser)',self.lL), ('MPI photons to ionize (l anti-Stokes)',self.lA),
                    ('zstart',self.zstart),('zend',self.zend),('zlen',self.zlen),('dz',self.dz),
                    ('tstart',self.tstart),('tend',self.tend),('tlen',self.tlen),('dt',self.dt),
                    ('rstart',self.rstart),('rend',self.rend),('rlen',self.rlen),('dr',self.dr),('Error threshold',self.errorThreshold)]
        for pair in textPairs: self.log(pair[0]+': '+self.display(pair[1]))
        self.log('Numerical energy in Stokes grid at start: '+self.display(self.getEnS())+" J")
        self.log('Numerical energy in Laser grid at start: '+self.display(self.getEnL())+" J")
        self.log('Numerical energy in anti-Stokes grid at start: '+self.display(self.getEnA())+" J")
        self.logSave('log_start')

    def boundaryConds(self):
        dr0 = 0. # Reflecting boundary at r=0
        self.AS[:,0] = 1./3*(-2*self.dr*dr0+4.*self.AS[:,1]-self.AS[:,2]) 
        self.AL[:,0] = 1./3*(-2*self.dr*dr0+4.*self.AL[:,1]-self.AL[:,2])
        self.AA[:,0] = 1./3*(-2*self.dr*dr0+4.*self.AA[:,1]-self.AA[:,2])
        return

    def calculateIonization(self):
 #       if self.ionMethod == 'MPI':
        calcNeMPI((self.AS,self.AL,self.AA), # calculate electron density
                  (self.WMPI_NH2O,self.WMPI_NH2O_S,self.WMPI_NH2O_L,self.WMPI_NH2O_A),
                  self.ve, self.ne, self.te, # comment GMP: added array te
                  (self.dt, self.eta, self.wS, self.wL, self.wA, self.nS, self.nL, self.nA, self.e0, self.c, self.NH2O, self.lS,self.lL,self.lA, self.IMPI, self.veC, self.viC)
        )
        if self.tlen==1:
            self.ne = (self.WMPI_NH2O/self.eta)

        if self.cap_Ne:
            self.ne = np.min([self.ne,self.ne*0+self.NH2O],axis=0)

        if self.warn_critical:
            if np.sum(self.ne > self.neCrit) > 0:
                print('Electron density has exceeded the critical density at step {:d}'.format(self.step))
                print('Ne max = {:.4g}/cm^3, critical density = {:.4g}/cm^3 (calculated for the anti-Stokes beam)'.format(np.max(self.ne)/1e6,self.neCrit/1e6))
                print('Exiting now')
                quit()
            
    def move(self): # Advance the simulation by one z step.
        self.boundaryConds() # Apply boundary conditions
        if self.step == 0: # Do this only on the first step
            self.constants_gen() # initialize all constants for the integration
            self.logInit() # write a short log to file

            self.RHSEuler = getRHS(self.constants,self.includes,euler=True)
            self.RHSCN = getRHS(self.constants,self.includes,euler=False)

        # Ionization
        if self.include_ionization:
            self.calculateIonization()

        if self.radial_filter:
            # Filter Ne to get rid of high-frequency numerical instabilities
            #neold = self.ne.copy()

            if self.step%self.radial_filter_interval == 0 and self.step>0:
                if self.radial_filter_field == 'electron density':
                    self.ne = self.radialFilter(self.ne)
                elif self.radial_filter_field == 'electric field':
                    as1, as2 = np.real(self.AS),np.imag(self.AS)
                    self.AS = self.radialFilter(as1)+(1j*self.radialFilter(as2))
                    al1, al2 = np.real(self.AL),np.imag(self.AL)
                    self.AL = self.radialFilter(al1)+(1j*self.radialFilter(al2))
                    aa1, aa2 = np.real(self.AA),np.imag(self.AA)
                    self.AA = self.radialFilter(aa1)+(1j*self.radialFilter(aa2))
                else:
                    print("Must specify radial_filter_field as 'electron density' or 'electric field'")
                    quit()

        # Calculate exp(i*k*z) at this step
        eikz0 = np.exp(1.j*self.dk*self.z) * self.include_fwm
        eikzNext = np.exp(1.j*self.dk*(self.z+self.dz)) * self.include_fwm
        # Allocate temporary arrays for the calculation
        ASnext = np.zeros((self.tlen,self.rlen), dtype=complexType)
        ALnext = np.zeros((self.tlen,self.rlen), dtype=complexType)
        AAnext = np.zeros((self.tlen,self.rlen), dtype=complexType)
        dASdz0 = np.zeros((self.tlen,self.rlen), dtype=complexType)
        dALdz0 = np.zeros((self.tlen,self.rlen), dtype=complexType)
        dAAdz0 = np.zeros((self.tlen,self.rlen), dtype=complexType)

        # Get the first (Euler) guess for the right hand side of our PDEs, using only the current zstep information
        self.RHSEuler( # Calculate updated envelope fields with the Euler method
            self.AS,self.AL,self.AA, # previous step fields in 
            self.AS,self.AL,self.AA, # fields with which to calculate new dAdz
            dASdz0,dALdz0,dAAdz0, # dAdz in (not actually used for euler method)
            ASnext, ALnext, AAnext, # fields out
            dASdz0, dALdz0, dAAdz0, # dAdz out (save this to use later in the Crank-Nicolson algorithm)
            self.WMPI_NH2O_S, self.WMPI_NH2O_L, self.WMPI_NH2O_A, # WMPI in
            self.ne, self.ve, eikz0, self.dz # more stuff
            )
        error = 1e30
        j = 0 # Crank-nicolson iteration parameter
        while(error > self.errorThreshold): # Crank-Nicolson method. Iterate until it converges.
            error = self.RHSCN( # Calculate updated envelope fields with the Crank-Nicolson method
                self.AS,self.AL,self.AA, # previous step fields in
                ASnext,ALnext,AAnext, # previous Crank-Nicolson fields in (for calculating dAdz) 
                dASdz0,dALdz0,dAAdz0, # dAdz in (guessed from previous timestep with Euler method)
                ASnext, ALnext, AAnext, # fields out
                dASdz0, dALdz0, dAAdz0, # dAdz out (we don't actually save these in the Crank-Nicolson method)
                self.WMPI_NH2O_S, self.WMPI_NH2O_L, self.WMPI_NH2O_A, # WMPI in
                self.ne, self.ve, eikzNext, self.dz
                )
            # decrease zstep if it is converging too slowly
            if(j>self.iterMax):
                if self.adaptive_zstep:
                    self.dz = self.dz/2.
                    self.adaptive_zstep_last = self.step
                    self.printIfConsoleOutput('Reducing zstep on step ',self.step,'iteration',j,' New zstep is',self.dz)
                    return
                else:
                    print('Convergence failure on step ',self.step,'iteration',j,'zstep is',self.dz)
                    quit()
            j = j+1
        if self.adaptive_zstep:
            # increase zstep if it is converging too fast
            if ((j <= 2) and (self.dz<self.dzMax) and (self.step-self.adaptive_zstep_last >= 10)): 
                self.dz = self.dz*2.
                self.adaptive_zstep_last = self.step
                self.printIfConsoleOutput('Increasing zstep on step ',self.step,'iteration',j,' New zstep is',self.dz)
                self.printIfConsoleOutput('error',error)
            if (self.dz < self.dzMin):
                self.printIfConsoleOutput('Convergence failure on step ',self.step,'iteration',j,'zstep is',self.dz)
                self.printIfConsoleOutput('error',error)
                quit()

        self.CNcount += j # keep track of how many iterations to converge

        # print status update to console
        if self.console_logging_interval>0 and self.step%self.console_logging_interval==0:
            stepChunk = int(self.step/self.console_logging_interval)
            tnew = time.time()
            dt = tnew - self.stepTime
            self.meanStepTime = 0 if self.step==0 else self.meanStepTime * (stepChunk-1)/stepChunk + dt / stepChunk
            self.printIfConsoleOutput('step: {:d}, iteration: {:d}, error: {:.4E}, z: {:.6f}, dz:{:.3g}, work time:{:.3g}, mean work time:{:.4g}'.format(self.step, j, error, self.z, self.dz,dt, self.meanStepTime)#,'energy',"{:.4E}".format(self.getEn(self.getIL())+self.getEn(self.getIS())+self.getEn(self.getIA())),'dz',"{:.4E}".format(self.dz)
                )
            self.stepTime = tnew
            

        # Saving data
        saveRestartNow = (self.saveRestarts and self.step%self.saveRestartInterval==0 and self.step != 0)
        saveScalarsNow = (self.saveScalars and (
            (self.saveScalarsInterval>0 and self.step%self.saveScalarsInterval==0) or
            (self.saveScalarsZInterval>0 and (int(np.floor(self.z/self.saveScalarsZInterval)) > int(np.floor((self.z-self.dz)/self.saveScalarsZInterval))))
        ))
        save2DNow = (self.save2D and (
            (self.save2DInterval>0 and self.step%self.save2DInterval==0) or
            (self.save2DZInterval>0 and (int(np.floor(self.z/self.save2DZInterval)) > int(np.floor((self.z-self.dz)/self.save2DZInterval))))
        ))
        save1DNow = (self.save1D and (
            (self.save1DInterval>0 and self.step%self.save1DInterval==0) or
            (self.save1DZInterval>0 and (int(np.floor(self.z/self.save1DZInterval)) > int(np.floor((self.z-self.dz)/self.save1DZInterval))))
        ))
        
        if (saveRestartNow or saveScalarsNow or save2DNow or save1DNow):
            IS, IL, IA = self.getIS(), self.getIL(), self.getIA() # We only calculate these if we might need them later.
            IT = (IS+IL+IA)
            ISmax,ILmax,IAmax = np.max(IS),np.max(IL),np.max(IA)

            if saveRestartNow:
                self.filer.savePickle(self, 'Restart_{:0>6d}'.format(self.step))

            if saveScalarsNow:
                saveArr = []
                if 'Energy_T' in self.saveScalarsWhich: # Keep track of whether we already calculated these
                    EnS, EnL, EnA = False, False, False
                for s in self.saveScalarsWhich:
                    if s =='step': saveArr.append(self.step)
                    elif s =='z': saveArr.append(self.z)
                    elif s == 'Energy_S': 
                        EnS = self.getEn(IS)
                        saveArr.append(EnS)
                    elif s == 'Energy_L': 
                        EnL = self.getEn(IL)
                        saveArr.append(EnL)
                    elif s == 'Energy_A': 
                        EnA = self.getEn(IA)
                        saveArr.append(EnA)
                    elif s == 'Energy_T': # If we already calcualted the three parts, then don't recalculate them
                        if EnS != False and EnL != False and EnA != False: saveArr.append(EnS+EnL+EnA)
                        else: saveArr.append(self.getEn(IS)+self.getEn(IL)+self.getEn(IA))
                    elif s == 'FWHM_S':
                        saveArr.append(self.getFWHM(self.getFluence(IS)))
                    elif s == 'FWHM_L':
                        saveArr.append(self.getFWHM(self.getFluence(IL)))
                    elif s == 'FWHM_A':
                        saveArr.append(self.getFWHM(self.getFluence(IA)))
                    elif s == 'FWHM_T':
                        saveArr.append(self.getFWHM(self.getFluence(IT)))
                    elif s == 'RMSSize_S':
                        saveArr.append(self.getRMSSize(self.getFluence(IS)))
                    elif s == 'RMSSize_L':
                        saveArr.append(self.getRMSSize(self.getFluence(IL)))
                    elif s == 'RMSSize_A':
                        saveArr.append(self.getRMSSize(self.getFluence(IA)))
                    elif s == 'RMSSize_T':
                        saveArr.append(self.getRMSSize(self.getFluence(IT)))
                    elif s == "IS_max":
                        saveArr.append(ISmax)
                    elif s == "IL_max":
                        saveArr.append(ILmax)
                    elif s == "IA_max":
                        saveArr.append(IAmax)
                    elif s == "ES_max":
                        saveArr.append(self.getEField(ISmax,self.nS))
                    elif s == "EL_max":
                        saveArr.append(self.getEField(ILmax,self.nL))
                    elif s == "EA_max":
                        saveArr.append(self.getEField(IAmax,self.nA))
                    elif s == "Ne_max":
                        saveArr.append(np.max(self.ne))
                    elif s == "Te_max":
                        saveArr.append(np.max(self.te))  # comment GMP: added on 01/25/2023
                    else:
                        print('Save scalar option "'+s+'" is not supported.')
                        quit()
                self.filer.appendRow(self.scalarsFile,saveArr)

            # Save 1D
            if save1DNow:
                FS = np.sum(IS, axis=0)*self.dt
                FL = np.sum(IL, axis=0)*self.dt
                FA = np.sum(IA, axis=0)*self.dt
                Ne_max = np.max(self.ne,axis=0)
                saveDict = {'rrange':[self.rstart,self.rend],'rlen':self.rlen,'trange':[self.tstart,self.tend],'tlen':self.tlen,'z':self.z,'dz':self.dz}
                for s in self.save1DWhich:
                    if s =='FS': saveDict[s] = np.sum(IS, axis=0)*self.dt
                    elif s =='FL': saveDict[s] = np.sum(IL, axis=0)*self.dt
                    elif s =='FA': saveDict[s] = np.sum(IA, axis=0)*self.dt
                    elif s =='FT': saveDict[s] = np.sum(IT, axis=0)*self.dt
                    elif s =='Ne_end': saveDict[s] = self.ne[-1,:]
                    elif s =='Ne_max': saveDict[s] = np.max(self.ne,axis=0)
                    elif s =='PS': saveDict[s] = self.getP(IS)
                    elif s =='PL': saveDict[s] = self.getP(IL)
                    elif s =='PA': saveDict[s] = self.getP(IA)
                    elif s =='PT': saveDict[s] = self.getP(IT)
                    elif s =='IS_mid': saveDict[s] = IS[int(self.tlen/2),:]
                    elif s =='IL_mid': saveDict[s] = IL[int(self.tlen/2),:]
                    elif s =='IA_mid': saveDict[s] = IA[int(self.tlen/2),:]
                    elif s =='ES_mid': saveDict[s] = self.getEField(IS[int(self.tlen/2),:],self.nS)
                    elif s =='EL_mid': saveDict[s] = self.getEField(IL[int(self.tlen/2),:],self.nL)
                    elif s =='EA_mid': saveDict[s] = self.getEField(IA[int(self.tlen/2),:],self.nA)
                    elif s =='PhaseS_mid': saveDict[s] = np.angle(self.AS[int(self.tlen/2),:])
                    elif s =='PhaseL_mid': saveDict[s] = np.angle(self.AL[int(self.tlen/2),:])
                    elif s =='PhaseA_mid': saveDict[s] = np.angle(self.AA[int(self.tlen/2),:])
                    elif s =='HankelS_mean': saveDict[s] = np.sqrt(np.sum(np.abs(self.radialHankel(self.getES()))**2,axis=0))
                    elif s =='HankelL_mean': saveDict[s] = np.sqrt(np.sum(np.abs(self.radialHankel(self.getEL()))**2,axis=0))
                    elif s =='HankelA_mean': saveDict[s] = np.sqrt(np.sum(np.abs(self.radialHankel(self.getEA()))**2,axis=0))
                    else:
                        print('Save 1D option "'+s+'" is not supported.')
                        quit()
                self.filer.savePickle(saveDict, 'Data1D_{:06d}'.format(self.step))
                #if self.saveLineouts and self.step%self.save1DInterval==0:
                #self.filer.savePickle({'Rstart':self.rstart, 'Rend':self.rend, 'rlen':self.rlen, 'z':self.z, 'dz':self.dz,'FS':FS, 'FL':FL, 'FA':FA, 'Ne_end':Ne_end, 'step':self.step},
                #                      'Lineouts_'+str(self.step))

            # Save 2D fields
            if save2DNow:
                saveDict = {'rrange':[self.rstart,self.rend],'rlen':self.rlen,'trange':[self.tstart,self.tend],'tlen':self.tlen,'z':self.z}
                for s in self.save2DWhich:
                    if s =='IS': saveDict[s] = IS.astype(float)
                    elif s =='IL': saveDict[s] = IL.astype(float)
                    elif s =='IA': saveDict[s] = IA.astype(float)
                    elif s =='IT': saveDict[s] = IT.astype(float)
                    elif s =='AS': saveDict[s] = self.AS.astype(np.complex64)
                    elif s =='AL': saveDict[s] = self.AL.astype(np.complex64)
                    elif s =='AA': saveDict[s] = self.AA.astype(np.complex64)
                    elif s =='ES': saveDict[s] = self.getEFromA(self.AS).astype(np.complex64)
                    elif s =='EL': saveDict[s] = self.getEFromA(self.AL).astype(np.complex64)
                    elif s =='EA': saveDict[s] = self.getEFromA(self.AA).astype(np.complex64)
                    elif s =='Ne': saveDict[s] = self.ne.astype(float)
                    else:
                        print('Save 2D option "'+s+'" is not supported.')
                        quit()
                self.filer.savePickle(saveDict, 'Data2D_{:06d}'.format(self.step))

        if self.include_stokes and self.updateS:
            self.AS = ASnext
        if self.updateL:
            self.AL = ALnext
        if self.include_antistokes and self.updateA:
            self.AA = AAnext
        self.z += self.dz
        self.step += 1
        
    def finish(self):
        self.timerEnd = time.time()*1000.0

        IS = self.getIS()
        enS = self.getEn(IS)
        IL = self.getIL()
        enL = self.getEn(IL)
        IA = self.getIA()
        enA = self.getEn(IA)
        self.log('----Results----')
        self.log("Elapsed time during integration: " + str(self.timerEnd-self.timerStart) + " ms for "+str(self.step)+" steps")
        if self.console_logging_interval>0:
            self.log("Average time for {:d} steps: {:g}: ".format(self.console_logging_interval,self.meanStepTime))
        self.log("Average iterations per step: " + str(self.CNcount/self.step))
        self.log('Numerical energy in Stokes beam at end: '+self.display(enS)+" J")
        self.log('\t that is '+self.display(enS/self.energy)+" times the input energy")
        self.log('Numerical energy in Laser beam at end: '+self.display(enL)+" J")
        self.log('\t that is '+self.display(enL/self.energy)+" times the input energy")
        self.log('Numerical energy in anti-Stokes beam at end: '+self.display(enA)+" J")
        self.log('\t that is '+self.display(enA/self.energy)+" times the input energy")
        self.logSave('log_end')


    def run(self):
        while self.z < self.zend:
            self.move()
        self.finish()
