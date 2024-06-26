# The source code has been authored by federal and non-federal employees. To the extent that a federal employee is an author of a portion of this software or a derivative work thereof, no copyright is claimed by the United States Government, as represented by the Secretary of the Navy ("GOVERNMENT") under Title 17, U.S. Code. All Other Rights Reserved. To the extent that a non-federal employee is an author of a portion of this software or a derivative work thereof, the work was funded in whole or in part by the U.S. Government, and is, therefore, subject to the following license: The Government has unlimited rights to use, modify, reproduce, release, perform, display, or disclose the computer software and computer software documentation in whole or in part, in any manner and for any purpose whatsoever, and to have or authorize others to do so. Any other rights are reserved by the copyright owner.

# Neither the name of NRL or its contributors, nor any entity of the United States Government may be used to endorse or promote products derived from this software, nor does the inclusion of the NRL written and developed software directly or indirectly suggest NRL's or the United States Government's endorsement of this product. 

# THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/../')
from Integrator import Integrator 
from scipy.optimize import curve_fit
from Tests import Test



def fn(constants):
    Rbeam = 150e-6 # starting beam radius (1/e^2 intensity)
    params = {
        'include_plasma_refraction': False,
        'include_ionization': False,
        'include_energy_loss': False,
        'include_raman': False,
        'include_fwm': False,
        'include_kerr': False,
        'include_group_delay': False,
        'include_gvd': False,
        'adaptive_zstep': True,

        'pulse_length_fwhm': [1e10], # Temporal lengths of each pulse (in vacuum)
        'toffset': [0.], # Offset for the multi-pulses
        'efrac': [1.], # Energy fraction in each pulse
        'pulse_intensity_radius_e2': [Rbeam],
        'Ibackground': 1e0,
        'focal_length': 1e10, # collimated beam
        'zrange': [0., 0.4],
        'trange': [0., 1e-10],
        'tlen': 4,
        'rrange': [0., 1e-3], 
        'rlen': 200, 
        'energy': 0.01e-3, # Pulse energy in J
        'plot2D_rlim': [0,200e-6], # 2D plotting radial limits
        'plot1D_rlim': [0,200e-6], # 1D plotting radial limits
        'file_output': False,
        'plot_real_time': False,
        'ionization_method': 'IMPI',#'IMPI',#'PMPB',
        'radial_filter': False,
        'console_logging_interval': 0,
    }
    for key, val in constants.items(): # Add constants
        params[key] = val

    sim = Integrator(params)

    zarr = []
    rarr = []
    Iarr = []

    while (sim.z < sim.zend):
        sim.move()
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


    lambda_laser = sim.lLv/sim.nL
    rayleigh_range = Rbeam**2*np.pi/lambda_laser
    i_rayleigh = np.min(np.where(zarr > rayleigh_range)[0])
    Ractual = rarr[i_rayleigh]
    Rexpect = Rbeam * np.sqrt(2) # Beam radius should be sqrt(2) times initial at the rayleigh range.
    

    # fig, ax = plt.subplots(1, 2, figsize=(8,4), dpi=120)
    # ax = ax.flatten()
    # ax[0].plot(zarr,rarr, label='measured')
    # ax[0].plot(zarr,Rbeam*np.sqrt(1+(zarr/rayleigh_range)**2),label='analytical')
    # plt.legend()
    # ax[1].imshow(Iarr[:,:100].T, origin='lower', aspect='auto')
    # plt.show()


    tolerance = 0.05*Rexpect # We will allow 5% tolerance
    result = np.abs(Ractual - Rexpect) < tolerance
    message = "Found radius of {:.6} m compared to expected {:.6} m at rayleigh range".format(Ractual, Rexpect)
    return result, message
    
test = Test("Diffraction", fn)
