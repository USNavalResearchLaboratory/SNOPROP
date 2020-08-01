# The source code has been authored by federal and non-federal employees. To the extent that a federal employee is an author of a portion of this software or a derivative work thereof, no copyright is claimed by the United States Government, as represented by the Secretary of the Navy ("GOVERNMENT") under Title 17, U.S. Code. All Other Rights Reserved. To the extent that a non-federal employee is an author of a portion of this software or a derivative work thereof, the work was funded in whole or in part by the U.S. Government, and is, therefore, subject to the following license: The Government has unlimited rights to use, modify, reproduce, release, perform, display, or disclose the computer software and computer software documentation in whole or in part, in any manner and for any purpose whatsoever, and to have or authorize others to do so. Any other rights are reserved by the copyright owner.

# Neither the name of NRL or its contributors, nor any entity of the United States Government may be used to endorse or promote products derived from this software, nor does the inclusion of the NRL written and developed software directly or indirectly suggest NRL's or the United States Government's endorsement of this product. 

# THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

import numpy as np
import matplotlib.pyplot as plt
import sys, os
from .simulation import Simulation
from .Tests import Test
from scipy.optimize import curve_fit



def fn(constants):
    Rpulse = 1e-3 # starting beam radius (1/e^2 intensity)
    params = {
        'include_plasma_refraction': True,
        'include_ionization': False,
        'include_energy_loss': False,
        'include_raman': False,
        'include_fwm': False,
        'include_kerr': False,
        'include_group_delay': False,
        'include_gvd': False,
        'adaptive_zstep': True,

        'profile_L': {
            'pulse_length_fwhm': [1e10], # Temporal lengths of each pulse (in vacuum)
            'toffset': [0.], # Offset for the multi-pulses
            'efrac': [1.], # Energy fraction in each pulse
            'pulse_radius_e2': [Rpulse],
            'energy': 0.001e-3, # Pulse energy in J
            'focal_length': 1e10, # collimated beam
        },
        'IBackground': 1e0,
        'N0': 33.3679e27,
        'zrange': [0., 1.0],
        'trange': [0., 10e-10],
        'tlen': 4,
        'rrange': [0., 3e-3], 
        'rlen': 800, 
        'file_output': False,
        'radial_filter': False,
        'console_logging_interval': 0,
    }
    for key, val in constants.items(): # Add constants
        params[key] = val

    sim = Simulation(params)

    zarr = []
    rarr = []
    Iarr = []
    

    # Make a graded index plasma lens to test refraction
    dlens = 0.01
    fplasma = 0.5
    alpha = sim.nL/(2*dlens*fplasma)
    ne0 = alpha * 2*sim.nL*sim.wL**2 * sim.e0*sim.me/(sim.e**2)
    nearr = ne0*sim.rr**2 # Propagating through dlens distance of this plasma should focus a collimated beam at fplasma.

    
    def nFromNe(ne):
        return np.sqrt(1-ne*sim.e**2/(sim.me*sim.e0*sim.wL**2))
    narr = nFromNe(nearr)
    
    #fig, ax = plt.subplots(2)
    #ax[0].plot(sim.r0, narr[0,:])
    #ax[0].set_xlabel('Radius (m)')
    #ax[0].set_ylabel('n(r)')
    #ax[1].plot(sim.r0, nearr[0,:])
    #ax[1].set_xlabel('Radius (m)')
    #ax[1].set_ylabel('Ne(r)')
    #plt.show()




    sim.ne = nearr
    
    plasmaLens = True
    stride = 10 # only save the beam size every 10 steps.
    while (sim.z < sim.zend):
        if sim.z > dlens and plasmaLens == True:
            sim.ne *= 0.
            plasmaLens = False
        sim.move()

        if sim.step%stride == 0:
            FL = np.mean(sim.getIL(),axis=0)
            r0 = sim.r0
            try:
                fitRes, fitConf = curve_fit(lambda x,I0,w:I0*np.exp(-2*x**2/(w*np.abs(w))),r0,FL,p0=[FL[0],sim.r0[-1]/5])
            except:
                fitRes = [0.,0.]
            zarr.append(sim.z)

            rarr.append(fitRes[1])
            Iarr.append(np.mean(sim.getIL(), axis=0))
    zarr = np.array(zarr)
    rarr = np.array(rarr)
    Iarr = np.array(Iarr)

    #fig, ax = plt.subplots(1, 2, figsize=(8,4), dpi=120)
    #ax = ax.flatten()
    #plt.sca(ax[0])
    #plt.plot(zarr,rarr, label='measured')
    #plt.legend()
    #plt.sca(ax[1])
    #plt.imshow(Iarr[:,:100].T, origin='lower', aspect='auto')
    #plt.colorbar()
    #plt.show()


    expect = fplasma
    actual = zarr[np.where(rarr == np.min(rarr))[0][0]]
    tolerance = 0.05*actual # We will allow 5% tolerance
    result = np.abs(actual - expect) < tolerance
    message = "Found plasma lens focus of {:.6} m compared to expected {:.6} m".format(actual, expect)
    return result, message
    
test = Test("Plasma focusing", fn)

