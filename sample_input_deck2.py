import numpy as np
import snoprop

params = { # Now we will enter the parameter dictionary
    # Material parameters
    'wavelength': 355e-9, # Pump laser wavelength in vacuum
    'wV': 2*3.14159*1.019e14, # Vibrational frequency in water is 101.9 THz
    'material': 'custom', # We'll supply our own refractive indices for water
    'nS': 1.34927, # Stokes index of refraction
    'nL': 1.35721, # Laser index of refraction
    'nA': 1.36624, # Anti-Stokes index of refraction
    'nSg': 1.40398, # Stokes group index
    'nLg': 1.42694, # Laser group index
    'nAg': 1.45577, # Anti-Stokes group index
    'gvd_bS': 1.07e-25, # Stokes GVD beta
    'gvd_bL': 1.32e-25, # Laser GVD beta
    'gvd_bA': 1.62e-25, # Anti-Stokes GVD beta
    'Uion': 9.5, # Ionization energy (eV)
    'N0': 3.34e28, # Water molecular number density (1/m^3)
    'sigmaC': 3.1e-20, # collision cross section (m^2)
    'IMPI': 1.6e+19, # Characteristic multiphoton ionization intensity (W/m^2)
    'eta': 1/1e-12, # Electron reattachment rate (s^-1)
    'IBackground': 1e2, # Background intensity (W/m^2)
    'n2Kerr': 5.06e-20, # Kerr index (n = n0 + n2*I) (m^2/W)
    'n2Raman': -1.69e-20j, # Raman index (should be imaginary) (m^2/W)
    'effective_mass': 0.2,

    # Toggle model components
    'include_plasma_refraction': True, # Toggle plasma refraction
    'include_ionization': True, # Toggle ionization
    'include_energy_loss': True, # Toggle energy loss to ionization and heating
    'include_raman': True, # Toggle stimulated Raman scattering
    'include_fwm': True, # Toggle four-wave mixing
    'include_antistokes': True, # Toggle whether to model the anti-Stokes beam at all
    'include_kerr': True, # Toggle Kerr focusing
    'include_group_delay': True, # Toggle group delay
    'include_gvd': True, # Toggle group velocity dispersion
    'adaptive_zstep': True, # Toggle adaptive zstep
    'radial_filter': False, # Smooth the electron density at each step
    'warn_critical': False, # Don't warn me if the electron density becomes critical
    'dz': 2e-07,
    'dz_min': 2e-09,

    # Grid parameters
    'zrange': [0, 0.004], # Stop the simulation at 4 mm
    'trange': [-1.5e-11,1.5e-11], # box temporal size is 30 ps
    't_clip': 5e-12, # Cut off the temporal profile 5ps from the box edge
    'tlen': 120, # Number of cells in time
    'rrange': [0., 200e-6], # Radial boundary at 400 microns
    'rlen': 800, # Number of cells in radius
    'iter_max': 40, # Max iterations for C-N solver

    # Pulse profiles    
    'profile_L': {
        'pulse_length_fwhm': [5e-12], # Temporal lengths of each sub-pulse
        'toffset': [0], # Temporal offsets for the sub-pulses
        'efrac': [1], # Energy fraction in each sub-pulse
        'pulse_radius_half': [24e-06], # Half width half max of each sub-pulse
        'focal_length': 0.002, # Focus at 2 mm.
        'energy': 4000000*5e-12, # Pulse energy in J
    },

    # Data output
    'save_restart_interval': 0, # Restarts disabled
    'save_scalars_z_interval': 20e-06, # z interval at which to save scalars
    #'save_scalars_interval': 40, # simulation step interval at which to save scalars
    'save_scalars_which': [ # Select which scalars to save to file
        # Individual and total energies
        'Energy_S','Energy_L','Energy_A','Energy_T',
        # Individual and total beam spot sizes (FWHM)
        'FWHM_S','FWHM_L','FWHM_A','FWHM_T',
        # Individual and total beam spot sizes (RMS integrated)
        'RMSSize_S','RMSSize_L','RMSSize_A','RMSSize_T',
        # Max electron density, intensities, and E fields anywhere in box
        'Ne_max','Te_max','IS_max','IL_max','IA_max','ES_max','EL_max','EA_max', # comment GMP: added Te_max
    ],
    'save_1D_z_interval': 100e-6, # z interval at which to save 1D data
    'save_1D_which': [ # Select which 1D data to save to file
        # Radial max of electron density, radial intensities at temporal middle of pulse, and radial average power and fluence
        'Ne_max','IS_mid','IL_mid','IA_mid','PL','PS','PA','FL','FS','FA',
    ],
    'save_2D_z_interval': 400e-6, # z interval at which to save 2D data
    'save_2D_which': [ # Select which 2d data to save to file
        # Electron density and the three (complex) electric fields
        'Ne','ES','EL','EA',
    ],
}

sim = snoprop.Simulation(params)
sim.run()
