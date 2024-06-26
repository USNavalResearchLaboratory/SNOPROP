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
import time


def fn(constants):
    Rpulse = 60e-6 # starting beam radius (1/e^2 intensity)

    energy = 1e-3
    tpulse = 100e-12
    powerExpected = energy/tpulse
    # Now get the max expected intensity. Since Rpulse is the 1/e^2 intensity, I(r) = I0*exp(-2(r/Rpulse)**2),
    # so power = Integrate[I(r)*2*pi*r,{r,0,Infinity}] = I0*pi/2*Rpulse**2. Thus I0 = power*2/pi/Rpulse**2
    ImaxExpected = powerExpected*2/np.pi/Rpulse**2
    
    params = {
        'include_plasma_refraction': False,
        'include_ionization': True,
        'include_energy_loss': False,
        'include_raman': False,
        'include_fwm': False,
        'include_kerr': False,
        'include_group_delay': False,
        'include_gvd': False,
        'adaptive_zstep': False,

        'pulse_length_fwhm': [1e10], # temporally flat pulse
        'toffset': [0.], # Offset for the multi-pulses
        'efrac': [1.], # Energy fraction in each pulse
        'pulse_intensity_radius_e2': [Rpulse],
        'Ibackground': 1e0,
        'focal_length': 1e10, # collimated beam
        'zrange': [0., 1.0],
        'trange': [0., 100e-12],
        'tlen': 200,
        'rrange': [0., 5e-3], 
        'rlen': 200, 
        'energy': energy, # Pulse energy in J
        'plot2D_rlim': [0,200e-6], # 2D plotting radial limits
        'plot1D_rlim': [0,200e-6], # 1D plotting radial limits
        'file_output': False,
        'plot_real_time': False,
        'ionization_method': 'IMPI',#'IMPI',#'PMPB',
        'radial_filter': False,
        'console_logging_interval': 0,
        'dz': 1e-9,
    }
    for key, val in constants.items(): # Add constants
        params[key] = val


    # Equation from Bahman's paper predicts the following electron distribution as a function of time for constant field:
    # Ne(t) = NH2O*WMPI/(eta-vi)*(1-exp((vi-eta)*t))
    # Set up the sim
    sim = Integrator(params)
    sim.calculateIonization() # Calculate electron density
    ne = sim.ne
    AL_axis = np.abs(sim.AL[:,0])
    I_axis = sim.getIL()[:,0]
    ne_axis = sim.ne[:,0]
    t = sim.t0

    
    
    # Calculate expected Ne
    from math import factorial as fac
    v_rms = sim.e*np.sqrt(2)/sim.me*AL_axis/sim.wL
    nu_e = sim.NH2O*sim.sigmaC*v_rms
    nu_i = nu_e/params['Uion']*2*sim.e**2/sim.me*(AL_axis/sim.wL)**2
    WMPI = 2*np.pi/fac(sim.lL-1)*sim.wL*(I_axis/sim.IMPI)**sim.lL

    ne_expected = sim.NH2O*WMPI/(params['eta']-nu_i)*(1-np.exp((nu_i-params['eta'])*t))

    # plt.figure()
    # plt.plot(ne_axis)
    # plt.plot(ne_expected,'r--')
    # plt.show()

    deviation = np.mean((ne_expected - ne_axis)/np.mean(ne_expected))
    tolerance = 0.01
    passed = deviation < tolerance
    return passed, 'Calculated electron density to within {:.3f}% of theory'.format(deviation*100)
    
test = Test("Ionization", fn)

