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

# Here we will test whether four-wave mixing matches Boyd's formulas in 10.3.38 and 10.3.39 as well as verify the magic angle

def fn(constants):
    Rpulse = 1e-3 # starting beam radius (1/e^2 intensity)
    params = {
        'include_plasma_refraction': False,
        'include_ionization': False,
        'include_energy_loss': False,
        'include_raman': True,
        'include_fwm': True,
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
        'rrange': [0., 5e-3], 
        'rlen': 8000, 
        'energy': 0.001e-3, # Pulse energy in J
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


    # prepare to log messages
    log = []

    # Set up the sim
    sim = Integrator(params)


    # This function will find the difference between AS and AA with a large laser field background.
    # Both AL and AA are initially travelling down the z axis with AS at an angle (specified as the
    # parameter) off axis. If angle is close to the magic Stokes angle, AA will align with it and travel
    # at the anti-Stokes magic angle in the other direction. This only works if dr is small enough.
    def test_fwm_angle(stokesAngle, plot=False):
        # Let's manually set the fields
        sim.mesh_init() # Reset simulation
        Rcenter = params['rrange'][-1]*.5 # Cut off our simulations before they get to the end of the box to avoid bad results
        Rwidth = params['rrange'][-1]*.4
        tlen, rlen = params['tlen'], params['rlen']
        radial_profile = np.exp(-((sim.rr-Rcenter)/Rwidth)**10) # Center the flat pulse at a non-zero radius (avoid the r=0 and r=end boundaries)
        #stokesAngle = .03932
        stokes_angle_profile = np.exp(-1.j*sim.kS*np.sin(stokesAngle)*sim.rr) # Profile to make the stokes wave travel off axis at angle stokesAngle
        ALmag = 5e8
        ASmag = 1e3
        AAmag = 1e3
        sim.AS = ASmag * radial_profile * stokes_angle_profile
        sim.AL = ALmag * radial_profile
        sim.AA = AAmag * radial_profile
        #print(np.angle(sim.AS[0,int(rlen/2)+1]/sim.AS[0,int(rlen/2)]))

        alphaS = 3j*sim.wS*sim.xRS*ALmag**2/(sim.nS*sim.c) # Predicted Stokes growth rate from Boyd 10.3.39a
        distS = np.real(1/alphaS) # Growth distance for 1 e-folding
        growth_factor = np.log((ALmag*.01)/ASmag) # Let the waves propagate for this many e-foldings of the Stokes wave growth.
        zend = growth_factor*distS # Growth distance

        stepsPerWavelength = 10
        dzMax = np.abs(2*np.pi/sim.dk) / stepsPerWavelength
        if sim.dz > dzMax:
            sim.dz = dzMax



        # Leftover code to plot the evolution of a four-wave mixing beam
        if plot:
            fig, ax = plt.subplots(figsize=(6,6), dpi=120)
            r = sim.r0
            d1, d2, d3 = np.abs([sim.AS[0,:], sim.AL[0,:], sim.AA[0,:]])
            l1, = ax.semilogy(r, d1,label='AS')
            l2, = ax.semilogy(r, d2,label='AL')
            l3, = ax.semilogy(r, d3,label='AA')
            plt.legend()
            plt.ylim(1e-2,ALmag*2)
            plt.show(block=False)
            fig.canvas.draw()
            while sim.z < zend:
                #print('step'+str(sim.step))
                sim.move()
                if sim.z > .20 or sim.step % 50 == 0:
                    d1, d2, d3 = np.abs([sim.AS[0,:], sim.AL[0,:], sim.AA[0,:]])
                    l1.set_ydata(d1)
                    l2.set_ydata(d2)
                    l3.set_ydata(d3)
                    ax.set_title('z={:8f}, AAphase={:4f}'.format(sim.z,np.angle(sim.AA[0,int(rlen/2)])))
                    plt.pause(.01)
                    fig.canvas.draw()
                    time.sleep(.1)
            plt.close(fig)
        else:
            while sim.z < zend:
                sim.move()
        r1, r2 = int(rlen*.4), int(rlen*.6) # index bounds in which to average the fields
        ASchunk, AAchunk = sim.AS[0,r1:r2], sim.AA[0,r1:r2]
        ASavg, AAavg = np.mean(np.abs(ASchunk)), np.mean(np.abs(AAchunk))

        return AAavg, ASavg, np.mean(np.angle(AAchunk[1:] / AAchunk[:-1])/(sim.kA*sim.dr)) # The last item here is the average angle at which the AA beam is propagating (the antiStokes angle)


        
#    for angle in np.linspace(sim.stokesAngle*.6,sim.stokesAngle*1.,3):
#        print(angle)
#        test_fwm_angle(angle)
#    return True, ''
        
    
    # We will test a series of angles near the expected magic angle. 
    expected = sim.stokesAngle
    angles = np.linspace(expected-.01, expected+.01,30)
    results = []
    for a in angles:
        #print('testing angle '+str(a))
        results.append(test_fwm_angle(a))
    ASarr = [r[0] for r in results]
    def upside_down_gaussian(x,y0,depth,x0,width):
        return y0 - depth*np.exp(-((x-x0)/width)**2)
    fit_res, pcov = curve_fit(upside_down_gaussian, angles, ASarr,
              p0=[np.mean(ASarr), max(ASarr)-min(ASarr), angles[np.where(ASarr==min(ASarr))[0][0]], (angles[-1]-angles[0])/10])
    magicAngle = fit_res[2]
    

    tolerance = 0.03 # 3% tolerance in magic angle.
    measured = magicAngle
    error = np.abs((measured - expected)/expected)
    result = error < tolerance
    return result, "Magic angle for the Stokes wave is approx {:.4f}, which is {:.3f}% different from the theoretical {:.4f}".format(measured, error/expected*100, expected)



    # # Get final and predicted fields

    # quit()

test = Test("Four-wave mixing", fn)

