# The source code has been authored by federal and non-federal employees. To the extent that a federal employee is an author of a portion of this software or a derivative work thereof, no copyright is claimed by the United States Government, as represented by the Secretary of the Navy ("GOVERNMENT") under Title 17, U.S. Code. All Other Rights Reserved. To the extent that a non-federal employee is an author of a portion of this software or a derivative work thereof, the work was funded in whole or in part by the U.S. Government, and is, therefore, subject to the following license: The Government has unlimited rights to use, modify, reproduce, release, perform, display, or disclose the computer software and computer software documentation in whole or in part, in any manner and for any purpose whatsoever, and to have or authorize others to do so. Any other rights are reserved by the copyright owner.

# Neither the name of NRL or its contributors, nor any entity of the United States Government may be used to endorse or promote products derived from this software, nor does the inclusion of the NRL written and developed software directly or indirectly suggest NRL's or the United States Government's endorsement of this product. 

# THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

import numpy as np
import matplotlib.pyplot as plt
import sys, os
from .simulation import Simulation
from .Tests import Test
from scipy.optimize import curve_fit


def getDivergenceAngle(params, frac=.25):
    # Run a simulation with parameters params and return the divergence angle of the resulting beam
    # frac is the last fraction of the beam to use when measuring divergence
    sim = Simulation(params)
    zarr = []
    rarr = []

    while (sim.z < sim.zend):
        sim.move()
        FL = np.mean(sim.getIL(),axis=0)
        r0 = sim.r0
        try:
            fitRes, fitConf = curve_fit(lambda x,I0,w:I0*np.exp(-x**2/(w*np.abs(w))),r0,FL,p0=[FL[0],sim.r0[-1]/5])
        except:
            fitRes = [0.,0.]
        zarr.append(sim.z)
        rarr.append(fitRes[1])
    zarr = np.array(zarr)
    rarr = np.array(rarr)
        
    imin = np.min(np.where(zarr > frac*sim.z)[0])
    zfit = zarr[imin:]
    rfit = rarr[imin:]
    dr = rfit[1:] - rfit[:-1]
    dz = zfit[1:] - zfit[:-1]
    dravg, dzavg = np.mean(dr), np.mean(dz)
    slope = dravg/dzavg
    angle = np.arctan2(dravg,dzavg)
    quality = np.std(dr/dz)

    #plt.figure()
    #plt.plot(zarr,rarr)
    #plt.ylabel('Pulse radius')
    #plt.xlabel('z')
    #plt.show()
    
    return angle, quality

def fn(constants):
    params = {
        'include_plasma_refraction': False,
        'include_ionization': False,
        'include_energy_loss': False,
        'include_raman': False,
        'include_fwm': False,
        'include_kerr': True,
        'include_group_delay': False,
        'include_gvd': False,
        'adaptive_zstep': True,

        'profile_L': {
            'pulse_length_fwhm': [1e10], # Temporal lengths of each pulse (in vacuum)
            'toffset': [0.], # Offset for the multi-pulses
            'efrac': [1.], # Energy fraction in each pulse
            'pulse_radius_e2': [1e-4*np.sqrt(2./np.log(2.))],
            'energy': 0.035e-3, # Pulse energy in J
            'focal_length': 1e10,
        },
        'IBackground': 1e0,
        'N0': 33.3679e27,
        'zrange': [0., 1.0],
        'trange': [0., 1e-10],
        'tlen': 4,
        'rrange': [0., 1e-3], 
        'rlen': 150, 
        'file_output': False,
        'radial_filter': False,
        'console_logging_interval': 0,
    }
    for key, val in constants.items(): # Add constants
        params[key] = val

    sim = Simulation(params)

    nL = sim.nL # get nLaser from the simulation calculation
    Pcrit = np.pi*.61**2*params['wavelength']**2/(8*nL*params['n2Kerr']) # According to Boyd 7.1.10
    #print('pcrit is '+str(Pcrit)+', frac is '+str(8e6/Pcrit))
    tau = (sim.tend - sim.tstart)
    Ecrit = tau * Pcrit



    # Narrow in on Ecrit. It should be close to Boyd's value.
    guess = 0.9*Ecrit
    step = 0.05*Ecrit
    previousAngleSign = 1
    params['profile_L']['energy'] = guess
    angle, quality = getDivergenceAngle(params)
    if angle<0:
        return False # Shouldn't be converging for our initial guess
    n = 0
    while step > 0.01*Ecrit: # Narrow in on Pcrit until we reach our step size limit.
        if angle*previousAngleSign<0: # Overshot Ecrit
            step /= 2
            previousAngleSign *= -1
            guess += previousAngleSign*step
            #print('flip! guess is '+str(guess) +' and angle is '+str(angle))
        else: # Need to keep marching
            guess += previousAngleSign*step
            #print('no flip. guess is '+str(guess) +' and angle is '+str(angle))
            params['profile_L']['energy'] = guess
        angle, quality = getDivergenceAngle(params)
        n += 1
        if n > 20: # Failed to find Ecrit.
            return False

    # Now "guess" contains the measured value of Ecrit.
    PcritMeasured = guess / tau
    PcritStep = step / tau

    tolerance = 0.05 # We will allow 5% tolerance compared to Boyd's number
    result = np.abs(PcritMeasured - Pcrit) / Pcrit < tolerance
    message = "Found Pcrit={:.2f} +- {:.2f} W compared to Boyd's predicted Pcrit={:.2f} W".format(
        PcritMeasured, 2*PcritStep, Pcrit)
    return result, message
    
test = Test("Critical power", fn)
