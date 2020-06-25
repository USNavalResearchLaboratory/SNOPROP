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


# Here we will test whether the Raman growth matches Boyd's theory in Eq 10.3.38-10.3.39

def fn(constants):
    Rpulse = 1e-3 # starting beam radius (1/e^2 intensity)
    params = {
        'include_plasma_refraction': False,
        'include_ionization': False,
        'include_energy_loss': False,
        'include_raman': True,
        'include_fwm': False,
        'include_antistokes': True,
        'include_kerr': False,
        'include_group_delay': False,
        'include_gvd': False,
        'adaptive_zstep': False,

        'pulse_length_fwhm': [1e10], # Temporal lengths of each pulse (in vacuum)
        'toffset': [0.], # Offset for the multi-pulses
        'efrac': [1.], # Energy fraction in each pulse
        'pulse_intensity_radius_e2': [Rpulse],
        'Ibackground': 1e0,
        'focal_length': 1e10, # collimated beam
        'zrange': [0., 1.0],
        'trange': [0., 10e-10],
        'tlen': 4,
        'rrange': [0., 3e-3], 
        'rlen': 800, 
        'energy': 0.001e-3, # Pulse energy in J
        'plot2D_rlim': [0,200e-6], # 2D plotting radial limits
        'plot1D_rlim': [0,200e-6], # 1D plotting radial limits
        'file_output': False,
        'plot_real_time': False,
        'ionization_method': 'IMPI',#'IMPI',#'PMPB',
        'radial_filter': False,
        'console_logging_interval': 20,
    }
    for key, val in constants.items(): # Add constants
        params[key] = val


    # prepare to log messages
    log = []

    # Set up the sim
    sim = Integrator(params)
    
    # Let's manually set the fields
    Rcenter = params['rrange'][-1]*.5 # Cut off our simulations before they get to the end of the box to avoid bad results
    Rwidth = params['rrange'][-1]*.4
    tlen, rlen = params['tlen'], params['rlen']
    radial_profile = np.exp(-((sim.rr-Rcenter)/Rwidth)**10) # Center the flat pulse at a non-zero radius (avoid the r=0 and r=end boundaries)
    ALmag = 1e8
    ASmag = 1e3
    AAmag = 1e3
    sim.AS = ASmag * radial_profile
    sim.AL = ALmag * radial_profile
    sim.AA = AAmag * radial_profile

    alphaS = 3j*sim.wS*sim.xRS*ALmag**2/(sim.nS*sim.c) # Predicted Stokes growth rate from Boyd 10.3.39a
    alphaA = 3j*sim.wA*sim.xRA*ALmag**2/(sim.nA*sim.c) # Predicted anti-Stokes growth rate from Boyd 10.3.39b

    
    distS = np.real(1/alphaS) # Growth distance for 1 e-folding
    zend = 4*distS # Let the waves propagate for 4 e-foldings of the Stokes wave growth.
    NstepsMin = 2000 # Take at least this many steps to integrate
    dzMax = zend/NstepsMin # dz should be at least this small
    if sim.dz > dzMax: sim.dz = dzMax
    while sim.z < zend:
        sim.move()
            
    # Get final and predicted fields
    ASend = np.abs(sim.AS[0,int(rlen/2)])
    ALend = np.abs(sim.AL[0,int(rlen/2)])
    AAend = np.abs(sim.AA[0,int(rlen/2)])
    ASpredicted = ASmag * np.exp(alphaS*zend) # Boyd 10.3.38a
    ALpredicted = ALmag
    AApredicted = AAmag * np.exp(alphaA*zend) # Boyd 10.3.38b

    tolerance = 0.01 # Allow 1% error
    ASresult = np.abs((ASend - ASpredicted)/ASpredicted)
    ALresult = np.abs((ALend - ALpredicted)/ALpredicted)
    AAresult = np.abs((AAend - AApredicted)/AApredicted)
    if not all([ASresult < tolerance,ALresult < tolerance,AAresult < tolerance]):
        return False, "Stokes growth rate off by {:f}%, anti-Stokes by {:f}%".format(ASresult*100,AAresult*100)
    log.append("Laser-to-Stokes wave Stokes Raman growth matches theory within {:f}%, anti-Stokes within {:f}%".format(ASresult*100,AAresult*100))


        


    

    
    # Now test anti-Stokes Raman scattering from the Stokes beam to the laser and the Raman scattering from the anti-Stokes beam to the laser
    sim.mesh_init() # Reset the simulation
    
    # Let's manually set the fields
    ASmag = 1e8
    ALmag = 1e3
    AAmag = 1e3
    sim.AS = ASmag * radial_profile
    sim.AL = ALmag * radial_profile
    sim.AA = AAmag * radial_profile

    alphaL = 3j*sim.wL*sim.xRA*ASmag**2/(sim.nL*sim.c) # Predicted anti-Stokes growth rate from Boyd 10.3.39a
    
    distL = np.real(-1/alphaL) # Growth distance for 1 e-folding
    zend = 4*distL # Let the waves propagate for 4 e-foldings of the Laser decay.
    NstepsMin = 2000 # Take at least this many steps to integrate
    dzMax = zend/NstepsMin # dz should be at least this small
    if sim.dz > dzMax: sim.dz = dzMax
    while sim.z < zend:
        sim.move()

    # Get final and predicted fields
    ASend = np.abs(sim.AS[0,int(rlen/2)])
    ALend = np.abs(sim.AL[0,int(rlen/2)])
    AAend = np.abs(sim.AA[0,int(rlen/2)])
    ASpredicted = ASmag
    ALpredicted = ALmag * np.exp(alphaL*zend) # Boyd 10.3.38b
    AApredicted = AAmag

    tolerance = 0.01 # Allow 1% error
    ASresult = np.abs((ASend - ASpredicted)/ASpredicted)
    ALresult = np.abs((ALend - ALpredicted)/ALpredicted)
    AAresult = np.abs((AAend - AApredicted)/AApredicted)
    if not all([ASresult < tolerance,ALresult < tolerance,AAresult < tolerance]):
        return False, "Stokes-beam-to-laser anti-stokes Raman scattering rate off by {:f}%".format(ALresult*100)
    log.append("Stokes-beam-to-laser anti-stokes Raman scattering rate matches theory within {:f}%".format(ALresult*100))





    # Now test anti-Stokes Raman scattering from the Stokes beam to the laser and the Raman scattering from the anti-Stokes beam to the laser
    sim.mesh_init() # Reset the simulation
    
    # Let's manually set the fields
    ASmag = 1e3
    ALmag = 1e3
    AAmag = 1e8
    sim.AS = ASmag * radial_profile
    sim.AL = ALmag * radial_profile
    sim.AA = AAmag * radial_profile

    alphaL = 3j*sim.wL*sim.xRS*AAmag**2/(sim.nL*sim.c) # Predicted Stokes growth rate from Boyd 10.3.39a
    
    distL = np.real(1/alphaL) # Growth distance for 1 e-folding
    zend = 4*distL # Let the waves propagate for 4 e-foldings of the Laser decay.
    NstepsMin = 2000 # Take at least this many steps to integrate
    dzMax = zend/NstepsMin # dz should be at least this small
    if sim.dz > dzMax: sim.dz = dzMax
    while sim.z < zend:
        sim.move()

    # Get final and predicted fields
    ASend = np.abs(sim.AS[0,int(rlen/2)])
    ALend = np.abs(sim.AL[0,int(rlen/2)])
    AAend = np.abs(sim.AA[0,int(rlen/2)])
    ASpredicted = ASmag
    ALpredicted = ALmag * np.exp(alphaL*zend) # Boyd 10.3.38b
    AApredicted = AAmag

    tolerance = 0.01 # Allow 1% error
    ASresult = np.abs((ASend - ASpredicted)/ASpredicted)
    ALresult = np.abs((ALend - ALpredicted)/ALpredicted)
    AAresult = np.abs((AAend - AApredicted)/AApredicted)
    if not all([ASresult < tolerance,ALresult < tolerance,AAresult < tolerance]):
        return False, "AntiStokes-beam-to-laser Stokes Raman scattering rate off by {:f}%".format(ALresult*100)
    log.append("AntiStokes-beam-to-laser Stokes Raman scattering rate matches theory within {:f}%".format(ALresult*100))

    return True, '\n'.join(log)
    
test = Test("Raman", fn)
