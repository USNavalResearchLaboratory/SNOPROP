import numpy as np
import snoprop
import multiprocessing as mp
import os
import time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def run_pinned_simulation(args):
    """
    Worker function that pins itself to specific CPU cores before running SNOPROP.
    """
    energy_uJ, assigned_cores, is_p_core_run = args
    
    # 1. PIN THIS PROCESS TO SPECIFIC CORES ONLY
    pid = os.getpid()
    
    if HAS_PSUTIL:
        # Cross-platform affinity setting (Required for Windows)
        p = psutil.Process(pid)
        p.cpu_affinity(assigned_cores)
    elif hasattr(os, 'sched_setaffinity'):
        # Native Linux affinity setting
        os.sched_setaffinity(pid, assigned_cores)
        
    # 2. RESTRICT NUMBA THREADS TO MATCH ASSIGNED CORES
    # We must restrict Numba so it doesn't spawn more threads than the cores we just pinned it to!
    num_threads = str(len(assigned_cores))
    os.environ["NUMBA_NUM_THREADS"] = num_threads
    os.environ["OMP_NUM_THREADS"] = num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = num_threads
    os.environ["MKL_NUM_THREADS"] = num_threads
    
    # 3. SET UP ISOLATED DIRECTORY
    core_type = "P-Core" if is_p_core_run else "E-Core"
    sim_dir = f"sim_E_{energy_uJ}uJ"
    os.makedirs(sim_dir, exist_ok=True)
    original_cwd = os.getcwd()
    
    try:
        os.chdir(sim_dir)
        
        pi=np.pi                                             # pi
        nm=1.0e-9                                            # 1 nm
        mkm=1.0e-6                                           # 1 micron
        cm=1.0e-2                                            # 1 cm
        fs=1.0e-15                                           # 1 fs
        ps=1.0e-12                                           # 1 ps
        mJ=1.0e-3                                            # 1 mJ
        microJ=1.0e-6                                        # 1 microJ
        
        # SNOPROP 2.0 Input Deck
        params = {
            # Material parameters
            'wavelength': 800.0*nm,                          # Pump laser wavelength in vacuum
            'wV': 2.0*pi*1.019e14,                           # Vibrational frequency in water is 101.9 THz
            'material': 'custom',                            # We'll supply our own refractive indices for water
            'nS': 1.34927,                                   # Stokes index of refraction
            'nL': 1.35721,                                   # Laser index of refraction
            'nA': 1.36624,                                   # Anti-Stokes index of refraction
            'nSg': 1.40398,                                  # Stokes group index
            'nLg': 1.42694,                                  # Laser group index
            'nAg': 1.45577,                                  # Anti-Stokes group index
            'gvd_bS': 1.07e-25,                              # Stokes GVD beta
            'gvd_bL': 1.32e-25,                              # Laser GVD beta
            'gvd_bA': 1.62e-25,                              # Anti-Stokes GVD beta
            'Uion': 9.5,                                     # Ionization energy (eV)
            'N0': 3.34e28,                                   # Water molecular number density (m^-3)
            'IBackground': 1e2,                              # Background intensity (W/m^2)
            'n2Kerr': 5.06e-20,                              # Kerr index (n = n0 + n2*I) (m^2/W) Chen-2020
            #'n2Kerr': 1.9e-20,                              # Kerr index (n = n0 + n2*I) (m^2/W) Winkler-2016
            'n2Raman': -1.69e-20j,                           # Raman index (should be imaginary) (m^2/W)
            'effective_mass': 0.5,                           # effective mass of electron in conduction band
        
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
            'dz': 20.0*nm,                                   # spatial step dz
            'dz_min': 2.0*nm,                                # minimum dz
            'zrange': [0,41.0*mkm],                          # comp. box length
            'trange': [-400.0*fs,400.0*fs],                  # box temporal size
            't_clip': 5.0*fs,                                # Cut off the temporal profile from the box edge
            'tlen': 400,                                     # Number of cells in time
            'rrange': [0.0,200.0*mkm],                       # Radial boundary
            'rlen': 200,                                     # Number of cells in radius
            'iter_max': 12,                                  # Max iterations for C-N solver
        
            # Pulse profiles    
            'profile_L': {
                'pulse_length_fwhm': [35.0*fs],              # Temporal lengths of each sub-pulse
                'toffset': [0],                              # Temporal offsets for the sub-pulses
                'efrac': [1],                                # Energy fraction in each sub-pulse
                'pulse_radius_half': [20.0*mkm],             # Half width half max of each sub-pulse
                'focal_length': 0.30,                        # Focus
                'energy': energy_uJ * microJ,                # Pulse energy as DYNAMIC PARAMETER
            },
        
            # Data output
            'save_restart_interval': 0,                      # Restarts disabled
            'save_scalars_z_interval': 0.2*mkm,              # z interval at which to save scalars
            #'save_scalars_interval': 40,                    # simulation step interval at which to save scalars
            'save_scalars_which': [                          # Select which scalars to save to file
                'Energy_S','Energy_L','Energy_A','Energy_T', # Individual and total energies
                'FWHM_S','FWHM_L','FWHM_A','FWHM_T',         # Individual and total beam spot sizes (FWHM)
                'RMSSize_S','RMSSize_L','RMSSize_A','RMSSize_T', # Individual and total beam spot sizes (RMS integrated)
                'Ne_max','Te_max',                           # max electron density anywhere in box
                'IS_max','IL_max','IA_max',                  # max intensities anywhere in box
                'ES_max','EL_max','EA_max',                  # max and E fields anywhere in box
            ],
            'save_1D_z_interval': 10.0*mkm,                  # z interval at which to save 1D data
            'save_1D_which': [                               # Select which 1D data to save to file
                # Radial max of electron density, radial intensities at temporal middle of pulse, and radial average power and fluence
                'Ne_max','IS_mid','IL_mid','IA_mid','PL','PS','PA','FL','FS','FA',
            ],
            'save_2D_z_interval': 10.0*mkm,                  # z interval at which to save 2D data
            'save_2D_which': [                               # Select which 2d data to save to file
                'Ne','ES','EL','EA',                         # Electron density and the three (complex) electric fields
            ],

        }
        
        # We must override FileIO to ensure it builds data/ and logs/ INSIDE the isolated dir
        original_init = snoprop.FileIO.__init__
        def custom_init(self):
            self.folder = ""
            self.dataDir = self.folder + "data/"
            self.logDir = self.folder + "logs/"
            self.simFolders = [self.dataDir, self.logDir]
            for f in self.simFolders: os.makedirs(f, exist_ok=True)
        snoprop.FileIO.__init__ = custom_init
        
        sim = snoprop.Simulation(params)
        sim.run()
        
        snoprop.FileIO.__init__ = original_init
        result_msg = f"✅ Success: {energy_uJ} uJ completed on {core_type}s {assigned_cores}."
        
    except Exception as e:
        result_msg = f"❌ Failed: {energy_uJ} uJ encountered error: {e}"
        
    finally:
        os.chdir(original_cwd)
        
    return result_msg


def build_hardware_topology():
    """
    Maps the logical cores of an Intel i9-14900K into separate P-Core and E-Core pools.
    Standard ACPI scheduling: Cores 0-15 are P-Cores, Cores 16-31 are E-Cores.
    """
    p_cores = list(range(0, 16))
    e_cores = list(range(16, 32))
    return p_cores, e_cores


if __name__ == '__main__':
    # 1. Define your energy scan queue (e.g., 20 scans)
    energy_scan_list = np.linspace(1.0, 22.0, 22).round(1).tolist()
    
    # 2. Topology and Limits
    MAX_THREADS_PER_SIM = 4  # Each simulation gets 4 cores (adjust up/down as needed)
    
    p_cores, e_cores = build_hardware_topology()
    worker_blocks = []
    
    # Slice P-Cores into blocks of 4
    for i in range(0, len(p_cores), MAX_THREADS_PER_SIM):
        block = p_cores[i:i+MAX_THREADS_PER_SIM]
        if len(block) == MAX_THREADS_PER_SIM:
            worker_blocks.append({'cores': block, 'is_p_core': True})
            
    # Slice E-Cores into blocks of 4
    for i in range(0, len(e_cores), MAX_THREADS_PER_SIM):
        block = e_cores[i:i+MAX_THREADS_PER_SIM]
        if len(block) == MAX_THREADS_PER_SIM:
            worker_blocks.append({'cores': block, 'is_p_core': False})
            
    num_concurrent_workers = len(worker_blocks) # 8 concurrent workers total (4 P-blocks, 4 E-blocks)
    
    print("==================================================")
    print(f"🚀 Advanced Hardware-Affinity SNOPROP Scheduler")
    print(f"CPU: Intel i9-14900K (32 logical cores mapped)")
    print(f"Thread Limit: {MAX_THREADS_PER_SIM} cores per simulation")
    print(f"Concurrent Worker Slots: {num_concurrent_workers}")
    print(f"Total simulations in queue: {len(energy_scan_list)}")
    print("==================================================\n")
    
    # 3. Create the queue arguments
    task_queue = []
    for i, energy in enumerate(energy_scan_list):
        # Round-robin assignment of tasks to specific hardware blocks
        assigned_block = worker_blocks[i % num_concurrent_workers]
        task_queue.append((energy, assigned_block['cores'], assigned_block['is_p_core']))
    
    # 4. Execute the Queue using mp.Pool
    mp.set_start_method('spawn')
    
    start_time = time.time()
    with mp.Pool(processes=num_concurrent_workers) as pool:
        # pool.map automatically dispenses tasks to available workers as they finish
        results = pool.map(run_pinned_simulation, task_queue)
        
    print("\n✅ All queued simulations complete!")
    print(f"⏱️ Total Wall Time: {(time.time() - start_time)/60:.2f} minutes.")
    for res in results:
        print("  -", res)
