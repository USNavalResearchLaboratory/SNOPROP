import numpy as np
import snoprop

# Before specifying the parameter dictionary,
# we will define a custom radial profile for the Stokes pulse
r0 = .3e-3 # Annulus major diameter in m
rwidth = .02e-3 # Annulus minor width in m
def radialProfile(r):
    return np.exp(-(r-r0)**2/(2*rwidth**2))

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
    'Uion': 9.0, # Ionization energy (eV)
    'N0': 3.33679e28, # Water molecular number density (1/m^3)
    'sigmaC': 3e-20, # collision cross section (m^2)
    'IMPI': 1.84e18, # Characteristic multiphoton ionization intensity (W/m^2)
    'eta': 1/1e-12, # Electron reattachment rate (s^-1)
    'IBackground': 1e2, # Background intensity (W/m^2)
    'n2Kerr': 5e-20, # Kerr index (n = n0 + n2*I) (m^2/W)
    'n2Raman': -1.7e-20j, # Raman index (should be imaginary) (m^2/W)


    # Toggle model components
    'include_plasma_refraction': True, # Toggle plasma refraction
    'include_ionization': True, # Toggle ionization
    'include_energy_loss': True, # Toggle energy loss to ionization and heating
    'include_raman': True, # Toggle stimulated Raman scattering
    'include_fwm': True, # Toggle four-wave mixing
    'include_kerr': True, # Toggle Kerr focusing
    'include_group_delay': True, # Toggle group delay
    'include_gvd': False, # Toggle group velocity dispersion
    'adaptive_zstep': True, # Toggle adaptive zstep
    'radial_filter': True, # Smooth the electron density at each step

    # Grid parameters
    'zrange': [0, .011], # Stop the simulation at 1cm
    'trange': [-25e-12,25e-12], # simulate a 40 ps box
    't_clip': 5e-12, # Cut off the temporal profile 5ps from the box edge
    'tlen': 200, # Number of cells in time
    'rrange': [0., 1e-3], # Radial boundary at 1 mm
    'rlen': 6000, # Number of cells in radius

    # Pulse profiles    
    'profile_L': {
        'pulse_length_fwhm': [10e-12], # Temporal lengths of each pulse
        'toffset': [0], # Offsets for the multi-pulses
        'efrac': [1], # Energy fraction in each pulse
        'pulse_radius_e2': [.1e-3], # Intensity to 1/e^2 radius
        'focal_length': 0.006,
        'energy': 6e-06, # Pulse energy in J
    },
    'profile_S': {
        'pulse_length_fwhm': [10e-12], # Temporal lengths of each pulse
        'toffset': [0], # Offsets for the multi-pulses
        'efrac': [1], # Energy fraction in each pulse
        'focal_length': 0.0055,
        'radial_func': radialProfile, # Custom annulus radial profile function
        'energy': 2e-06, # Pulse energy in J
    },

    # Data output
    'save_restart_interval': 0, # Restarts disabled
    'save_scalars_interval': 20, # Interval at which to save scalars
    'save_scalars_which': [ # Select which scalars to save to file
        # Individual and total energies
        'Energy_S','Energy_L','Energy_A','Energy_T',
        # Individual and total beam spot sizes (FWHM)
        'FWHM_S','FWHM_L','FWHM_A','FWHM_T',
        # Individual and total beam spot sizes (RMS integrated)
        'RMSSize_S','RMSSize_L','RMSSize_A','RMSSize_T' 
    ],
    'save_1D_interval': 100, # Interval at which to save 1D data
    'save_1D_which': [ # Select which 1D data to save to file
        # Power vs time, fluence vs radius, and final/peak electron densities
        'PS','PL','PA','FS','FL','FA','Ne_end','Ne_max', 
    ],
    'save_2D_interval': 9900, # Interval at which to save 2D data
    'save_2D_which': [ # Select which 2d data to save to file
        'IS','IL','IA','Ne', # Intensities and electron density
    ],
}

sim = snoprop.Simulation(params)
sim.run()
