import numpy as np
import snoprop

scale_factor=1.0

params = { # Now we will enter the parameter dictionary
    # Material parameters
    'wavelength': 785e-9, # Pump laser wavelength in vacuum
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
    'n2Kerr': 5.06e-20, # Kerr index (n = n0 + n2*I) (m^2/W) Chen-2020
    #'n2Kerr': 1.9e-20, # Kerr index (n = n0 + n2*I) (m^2/W) Winkler-2016
    'n2Raman': -1.69e-20j, # Raman index (should be imaginary) (m^2/W)
    'effective_mass': 0.2,

    # Toggle model components
    'include_plasma_refraction': True,               # Toggle plasma refraction
    'include_ionization': True,                      # Toggle ionization
    'include_energy_loss': True,                     # Toggle energy loss to ionization and heating
    'include_raman': True,                           # Toggle stimulated Raman scattering
    'include_fwm': True,                             # Toggle four-wave mixing
    'include_antistokes': True,                      # Toggle whether to model the anti-Stokes beam at all
    'include_kerr': True,                            # Toggle Kerr focusing
    'include_group_delay': True,                     # Toggle group delay
    'include_gvd': True,                             # Toggle group velocity dispersion
    'adaptive_zstep': True,                          # Toggle adaptive zstep
    'radial_filter': False,                          # Smooth the electron density at each step
    'warn_critical': False,                          # Don't warn me if the electron density becomes critical

    # Grid parameters
    'dz': 0.02e-06,                                  # (parameter changed by GMP)
    'dz_min': 1e-09,                                 # (parameter changed by GMP)
    'zrange': [0,120e-6/scale_factor],               # (parameter changed by GMP)
    'trange': [-300.0e-15,300.0e-15],                # box temporal size is 300 fs (parameter changed by GMP)
    't_clip': 5.0e-15,                               # Cut off the temporal profile 20 fs from the box edge
    'tlen': 300,                                     # Number of cells in time (parameter changed by GMP)
    'rrange': [0.0,25.0e-6/scale_factor],            # Radial boundary at 25 mkm, reduced by a factor of 4 (parameter changed by GMP)
    'rlen': int(250/scale_factor),                   # Number of cells in radius 250, reduced by a factor of "scale" (parameter changed by GMP)
    'iter_max': 12, # Max iterations for C-N solver

    # Pulse profiles    
    'profile_L': {
        'pulse_length_fwhm': [35.0e-15],             # Temporal lengths of each sub-pulse (parameter changed by GMP)
        'toffset': [0],                              # Temporal offsets for the sub-pulses (parameter left the same by GMP)
        'efrac': [1],                                # Energy fraction in each sub-pulse (parameter left the same by GMP)
        'pulse_radius_half': [7.0e-06/scale_factor], # Half width half max of each sub-pulse, reduced by a factor of 4 (parameter changed by GMP)
        'focal_length': 0.40/scale_factor,           # Focus at 40 cm, reduced by a factor of 4 (parameter changed by GMP)
        'energy': 1.5e-6,                            # Pulse energy in J (parameter changed by GMP)
    },

    # Data output
    'save_restart_interval': 0, # Restarts disabled
    'save_scalars_z_interval': 1.0e-06, # z interval at which to save scalars (parameter changed by GMP)
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
    'save_1D_z_interval': 1.0e-5, # z interval at which to save 1D data (parameter changed by GMP)
    'save_1D_which': [ # Select which 1D data to save to file
        # Radial max of electron density, radial intensities at temporal middle of pulse, and radial average power and fluence
        'Ne_max','IS_mid','IL_mid','IA_mid','PL','PS','PA','FL','FS','FA',
    ],
    'save_2D_z_interval': 1.0e-5, # z interval at which to save 2D data (parameter changed by GMP)
    'save_2D_which': [ # Select which 2d data to save to file
        # Electron density and the three (complex) electric fields
        'Ne','ES','EL','EA',
    ],
}

sim = snoprop.Simulation(params)
sim.run()
