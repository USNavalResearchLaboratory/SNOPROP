# Input guide

SNOPROP simulations are typically run using a user-specified python
input file that imports the `snoprop` package, defines the laser pulse
and medium, and calls the `.run()` method. A template input script reads
as follows.

    import snoprop
    params = { } # Define simulation parameters in here
    sim = snoprop.Simulation(params)
    sim.run()

Aside from any extra computations or setup the user wishes to do, the
bulk of a simulation script usually consists of defining the parameter
dictionary. The parameters fall into five main categories: material,
model, grid, pulse, and diagnostics. Each of these will be discussed in
the next sections.

# Material specification

The parameter dictionary must include material parameters for the homogeneous background medium.

The user may specify refractive indices in one of three methods. In the first method, the `material` parameter is set to `` 'custom' `` and the user may specify the relevant linear refractive indices $n_n$ (for $n \elem \{S,L,A\}$), group indices $n_{gn}$, and GVD parameters $\beta''_n$ for each beam in the input parameter dictionary.

In the second method, the user sets the `material` parameter to a python function which accepts a float (wavelength in m) and returns a float (index of refraction). The index of refraction, group indices, and GVD parameters for each beam are calculated from the refractive index data. With this method, the user can input custom refractive index data from tables or using the Sellmeier equation.

In the third method, the user may set `material` to `` `vacuum' `` which assumes a refractive index of unity.

| Key | Required? | Type | Description |
| ------ | ------ | ------ | ------ |
| `wavelength` | yes | float | Laser wavelength (m) |
| `material` | yes | string or function | Material name. May be set to `'vacuum'` to ignore all material effects, `'custom'` in which case several optional parameters for refractive indices, group indices, and group velocity dispersion parameters are required (see below), or a python function accepting as input a float (wavelength in m) and returning a float (refractive index). |
| `wV` | yes | float | Material vibration frequency (1/s) |
| `Uion` | yes | float | Material ionization energy (eV) |
| `sigmaC` | yes | float | Collision cross section (m$^2$) |
| `IMPI` | yes | float | Multiphoton ionization characteristic intensity (W/m$^2$) |
| `eta` | yes | float | Electron loss rate (1/s) |
| `IBackground` | yes | float | Initial background intensity for all fields (W/m^2). This can be overridden if `IBackground` is specified as an element of the pulse profiles `profile_S`, `profile_L`, `profile_A` (see Pulse profile section below). |
| `n2Kerr` | yes | float | Kerr index $n_2$ (m$^2$/W). This is defined as in $n = n_0 + n_2I$ with $I$ an intensity in [W/m$^2$] |
| `n2Raman` | yes | complex | Raman index; should be imaginary (m$^2$/W) |
| `effective_mass` | no | float | Effective electron mass (fraction of electron mass) (default 1.0) |
| `nS` | no | float | Stokes refractive index (req. if `material = 'custom'`) |
| `nL` | no | float | Laser refractive index (req. if `material = 'custom'`) |
| `nA` | no | float | Anti-Stokes refractive index (req. if `material = 'custom'`) |
| `nSg` | no | float | Stokes group index (req. if `material = 'custom'`) |
| `nLg` | no | float | Laser group index (req. if `material = 'custom'`) |
| `nAg` | no | float | Anti-Stokes group index (req. if `material = 'custom'}) |
| `gvd_bS` | no | float | Stokes GVD parameter (s$^2$/m) (req. if `material = 'custom'`) |
| `gvd_bL` | no | float | Laser GVD parameter (s$^2$/m) (req. if `material = 'custom'`) |
| `gvd_bA` | no | float | Anti-Stokes GVD parameter (s$^2$/m) (req. if `material = 'custom'`) |
| 'Ne_func' | no | function | May specify externally-generated electron density. The function should accept two floats (time in s and radius in m) and return an electron density float in 1/m$^3$. This option requires setting `include_ionization` to False |





# Model components
Individual model components such as SRS, four-wave mixing, GVD, ionization, and others can be enabled or disabled in the simulation input. By default, the code will only advance the Stokes and anti-Stokes fields if Raman scattering is enabled. Thus, a single-envelope simulation including only the laser field can be performed (with significantly shorter execution time) by disabling Raman scattering entirely.

By default, the code will only advance the Stokes and anti-Stokes fields if Raman scattering is enabled. Thus, a single-envelope simulation including only the laser field can be performed (with significantly shorter execution time) by disabling Raman scattering entirely.

Additional options in the solver can also be enabled here such as the adaptive $z$ step. If the adaptive $z$ step is enabled, then the solver automatically reduces the $z$ step size by half when a large number of iterations is required to meet the error threshold. If the solver converges in a single iteration (and the current $z$ step is less than the CFL condition requires), then the $z$ step size is doubled. If the error remains large after many iterations and the code has already reduced the $z$ step size by a factor of $10^4$, the solver will exit with a description of the convergence failure.

The code also features an optional radial filter which damps high frequencies in either the electron density or the electric fields and may be useful to suppress numerical instabilities (D. Potter, 'Computational Physics'. Wiley, Chester, UK, 1973). When filtering the electron density, the user should remember that it is calculated anew at every $z$-step and so it should be filtered at every $z$ step. However, filtering the electric field at every step can quickly lead to significant energy loss and so it should be used sparingly. The filter may operate in real space (as a Gaussian filter with standard deviation of 1 cell in the radial direction), or in frequency space (using a Hankel transform and smoothly damping the upper half of frequencies). The Hankel filter is weighted as $w(f) = -0.5\cos[\pi(0.5\cos(\pi f/F)+0.5)]+0.5$ for frequency $f$ with $F = 1/(2dr)$ being the maximum frequency. The Hankel filter may significantly reduce performance with a large number of grid points in $r$. More advanced filtering could be performed manually in the simulation script by interacting with the simulation object during execution (see Methods information below).



| Key | Required? | Type | Description |
| ------ | ------ | ------ | ------ |
| `include_plasma_refraction` | yes | bool | Toggle plasma refraction |
| `include_ionization` | yes | bool | Toggle ionization |
| `include_energy_loss` | yes | bool | Toggle laser energy loss to ionization and heating |
| `include_raman` | yes | bool | Toggle stimulated Raman scattering |
| `include_fwm` | yes | bool | Toggle four-wave mixing |
| `include_kerr` | yes | bool | Toggle Kerr focusing |
| `include_group_delay` | yes | bool | Toggle group delay |
| `include_gvd` | yes | bool | Toggle group velocity dispersion |
| `include_stokes` | no | bool | Toggle whether to update the Stokes field. Default is the value of `include_raman`. |
| `include_antistokes` | no | bool | Toggle whether to update the anti-Stokes field. Default is the value of `include_raman`. |
| `adaptive_zstep` | no | bool | Toggle adaptive $z$ step. Default is true. |
| `radial_filter` | no | bool | Toggle the radial filter. Default is false. |
| `radial_filter_interval` | no | int | Specify the interval (number of simulation steps) at which to apply the radial filter. Default is 1. |
| `radial_filter_type` | no | string | Specify the filter type: either 'gaussian' (real-space filter) or 'bessel' (frequency-space filter). Default is 'gaussian'. |
| `radial_filter_field` | no | string | Specify which field to apply the radial filter to. Options are 'electron density' and 'electric field'; currently, only one option is possible at a time. Default is 'electron density'. |









# Simulation grid

The simulation box size and resolution is specified for both the time $\tau$ and radius $r$ axes. In addition, the user specifies $z$ limits and may optionally specify a step size $dz$. By specifying only one cell in either radius or time, the user can perform one-dimensional simulations. In the 1D radial geometry, collisional ionization is not supported. If MPI ionization is enabled, then the electron density will be calculated as the steady state density where MPI ionization rate is exactly balanced by the electron loss rate $N_0 \eta$.


| Key | Required? | Type | Description |
| ------ | ------ | ------ | ------ |
| `zrange` | yes | [float] | Simulation z limits (m). Should be a 2-tuple with the first element 0. |
| `dz` | no | float | Manually specify the maximum simulation $z$ step size (m). If adaptive_zstep is disabled, then the $z$ step size will be fixed to this value for the duration of the simulation. |
| `trange` | yes | [float] | Simulation $\tau$ limits (s). Should be a 2-tuple. |
| `t_clip` | no | float | Temporal boundary width in which to clip the pulses (s). Useful to clip gaussian waveforms so they do not extend to the boundary where they can introduce unphysical boundary effects. |
| `tlen` | yes | int | Number of steps in $\tau$. |
| `rrange` | yes | [float] | Simulation r limits (m). Should be a 2-tuple with the first element 0. |
| `rlen` | yes | int | Number of steps in r. |






# Pulse profile

Each of the three beams (laser, Stokes, and anti-Stokes) may be initialized with a user-specified profile and unique background intensities. To input a laser profile, set the `profile_L` parameter in the main simulation dictionary equal to another dictionary containing the necessary pulse parameters as shown in Table \ref{tab:pulseprofile}. The laser profile (`profile_L`) is required, while the Stokes (`profile_S`) and anti-Stokes (`profile_A`) are optional and will default to the uniform background intensity specified globally in the `IBackground` parameter (see Table \ref{tab:mats}). 

The pulse profiles may be input in three ways:
1. Users may specify an arbitrary number of radially- and temporally-Gaussian pulses in a pulse train. In this case, each pulse is given a duration, temporal offset, spot size, and fraction of the total energy, and the whole pulse is given an energy, focal length, and optionally a background intensity.
2. Users may input the radial or temporal profile or both as an array of fluence vs $r$ or power vs $\tau$. In this case, the waveform resulting from a radial profile $F(r)$ and temporal profile $P(\tau)$ is simply $I(\tau,r) \propto F(r)P(\tau)$ scaled to reach the appropriate total beam energy. Even if both radial and temporal profiles are suppled, the user must still specify a pulse in the pulse train and associated energy and focal length.
3. Users may explicitly input the full 2D complex envelope fields $A_S$, $A_L$, or $A_A$ ordered with the first index being $\tau$ and the second $r$. In this case, no other options must be specified. If an optional energy is specified, the entire field will be scaled accordingly.

With each of these methods, if a background intensity is specified in the pulse parameters, this background will be added to the field after the pulse has already been scaled to the appropriate energy.
  
| Key | Required? | Type | Description |
| ------ | ------ | ------ | ------ |
| `pulse_length_fwhm` | yes | [float] | Intensity FWHM temporal lengths of each pulse in pulse train (s). |
| `toffset` | yes | [float] | Temporal offsets for each pulse in the pulse train (s). |
| `efrac` | yes | [float] | Fraction of total energy in each pulse in pulse train. |
| `pulse_radius_half` | yes | [float] | Spot radius to half max intensity for each pulse in pulse train (m). |
| `pulse_radius_e2` | no | [float] | Spot radius to 1/e$^2$ intensity for each pulse in pulse train (m) (may specify this parameter alternatively to \ttfamily{pulse_intensity_radius_half}) |
| `focal_length` | no | float | Focus distance (m). Must specify either this input or the `axicon_angle` input. |
| `axicon_angle` | no | float | Axicon angle for axicon-focused beam (rad). Must specify either this input or the `focal_length` input. |
| `energy` | no | float | Total energy of the entire pulse train (J). Must specify either this input or the `intensity` input. |
| `intensity` | no | float | Peak intensity of the pulse train (W/m$^2$). Must specify either this input or the `energy` input. |
| `radial_data` | no | [[float], [float]] | Radial data for the pulse profile. The first array is radius in m and the second is fluence (which will be normalized by the code to reach the specified pulse energy). |
| `temporal_data` | no | [[float], [float]] | Temporal data for the pulse profile. The first array is time in s and the second is power (which will be normalized by the code to reach the specified pulse energy). |
| `2D_data` | no | 2D complex | Spatiotemporal electric field envelope profile. The user may specify the entire electric field envelope as a 2D complex array where the first axis is time $\tau$ and the second axis is radius $r$. The resulting profile will be normalzed to the correct power and a background field added (if a background field has been specified either globablly or for this particular profile). The field will be normalized by the code to reach the specified pulse energy. |
| `update` | no | bool | Specify whether to update this field at every simulation timestep. Set to False to keep the field fixed during the simulation (for example, to run in the strong pump approximation). Default is True. |

# Diagnostics

When a simulation is initialized, the code attempts to create folders for the output files in the current working directory if file output is enabled. These folders are named `data/`, where all diagnostic data is saved, and `logs/`, where input and calculated parameters are saved at the start and end of each simulation. If these folders already exist, the code will inform the user and exit.

Various scalar, 1D, and full 2D diagnostics may be saved to disk at intervals specified by the user. All intervals are specified as number of integration steps. If the user wishes constant spacing between diagnostic saves, then they should not use the adaptive $z$ step feature.

Scalar diagnostics are saved in a single file called `scalars.csv`; new scalar data is appended to the end of this file as the simulation progresses. 1D data are saved as a new Python pickle (`.pckl`) file for each $z$-step which matches the specified interval with the prefix `Data1D_` followed by the step number. 2D data are also saved as `.pckl` files with the prefix `Data2D_` followed by the step number. The user may choose specific scalar, 1D and 2D diagnostics for specific beams; the options are detailed in the table.

Restart files may also be saved at user-specified intervals. The code simply saves the simulation object in a `.pckl` file in the `data/` directory with the prefix `Restart_` followed by the $z$-step. To use a restart file, the user should import the `pickle` and `snoprop` packages and then load the simulation object with (for example) `sim = pickle.load(open('data/Restart_001000.pckl','rb'))`. Then the simulation may be run with `sim.run()` as before.



| Key | Required? | Type | Description |
| ------ | ------ | ------ | ------ |
| `file_output` | no | bool | Toggle all file output including all diagnostics described in this section. Default is true (permit file output). |
| `console_logging_interval` | no | int | Specify interval at which to log simulation progress in the console; set to 0 to disable console logging. Default is 20. |% The program will log the step number, iteration count required for convergence, residual error, $z$ position, $z$ step size, and work time since last log compared to average work time. Set to 0 to disable all console output (except errors). Default is 20. 
| `save_restart_interval` | no | int | Interval in simulation steps at which to save restart files. Default is 0 (no saves). |
| `save_scalars_interval` | no | int | Interval in simulation steps at which to save scalar data. Default is 0 (no saves). |
| `save_scalars_z_interval` | no | int | Interval in $z$ distance (m) at which to save restart files. Default is 0 (no saves). |
| `save_scalars_which` | no | [string] | Select scalars to save to file. Currently supported options are the Stokes, laser, anti-Stokes, and total energies (`Energy_S`,`Energy_L`, `Energy_A`, `Energy_T`), FWHM spot size for the Stokes, laser, anti-Stokes, and total beams (`FWHM_S`, `FWHM_L`, `FWHM_A`, `FWHM_T`), the RMS spot sizes for the Stokes, laser, anti-Stokes, and total beams (`RMSSize_S`, `RMSSize_L`, `RMSSize_A`, `RMSSize_T`), max intensity for each beam (`IS_max`, `IL_max`, `IA_max`), max electric field strength for each beam (`ES_max`, 'EL_max`, `EA_max`), and max electron density (`Ne_max`). Simulation step and $z$ position are automatically saved as well. |
| `save_1D_interval` | no | int | Interval in simulation steps at which to save 1D profiles. Default is 0 (no saves). |
| `save_1D_z_interval` | no | int | Interval in $z$ distance (m) at which to save 1D profiles. Default is 0 (no saves). |
| `save_1D_which` | no | [string] | Select which 1D data to save to file. Currently supported options are the Stokes, laser, anti-Stokes, and total fluence vs $r$ (`FS`, `FL`, `FA`, `FT`), the Stokes, laser, anti-Stokes, and total beam powers vs $\tau$ (`PS`, `PL`, `PA`, `PT`), the intensity lineout along the temporal center of the beam vs $r$ (`IS_mid`, `IL_mid`, `IA_mid`), the electric field lineout along the temporal center of the beam vs $r$ (`ES_mid`, `EL_mid`, `EA_mid`), the electric field phase lineout along the temporal center of the beam vs $r$ (`PhaseS_mid`, `PhaseL_mid`, `PhaseA_mid`), the radial Hankel-filtered k-space beam spectra averaged over the temporal direction (`HankelS_mean`, `HankelL_mean`, `HankelA_mean`), and the peak (`Ne_max`) or final (`Ne_end`) electron density vs $r$. The $z$ position and axes information are also saved. |
| `save_2D_interval` | no | int | Interval in simulation steps at which to save 2D data. Default is 0 (no saves). |
| `save_2D_z_interval` | no | int | Interval in $z$ distance (m) at which to save 2D data. Default is 0 (no saves). |
| `save_2D_which` | no | [string] | Select which 2D data to save to file. Currently supported options are the Stokes, laser, anti-Stokes, and total intensity (`IS`, `IL`, `IA`, `IT`), Stokes, laser, and anti-Stokes electric fields (`ES`, `EL`, `EA`), Stokes, laser, and anti-Stokes envelope fields (`AS`, `AL`, `AA`), and electron density (`Ne`). The $z$ position as well as axes limits are also saved. |


    
# Interacting with the simulation

In addition to running the entire simulation via the `.run()` method, the user may also run it step-by-step with the `.move()` method. Many more public methods are implemented to extract simulation parameters and fields as well as manually edit fields using the simulation object. These methods are detailed below. Note that all 2D data types are Numpy arrays.

| Method | Input | Returns | Description |
| ------ | ------ | ------ | ------ |
| `.run()` |  |  | Run the simulation until $z$ reaches the end specified in the parameter dictionary. |
| `.move()` |  |  | Step the simulation forward by one $z$-step. |
| `.getZ()` |  | float | Get the current simulation distance $z$ (m). |
| `.getBeamParams()` | | dict | Returns a dictionary with the vacuum wavelenths, refractive indices, group indices, and GVD beta parameters for each beam. |
| `.getGrid()` | | dict | Returns a dictionary with the grid parameters in $r$, $\tau$, and $z$ as well as the 1D and 2D grids themselves. | 
| `.getEnS()` |  | float | Get the Stokes energy (J). |
| `.getEnL()` |  | float | Get the laser energy (J). |
| `.getEnA()` |  | float | Get the anti-Stokes energy (J). |
| `.getIS()` |  | 2D float | Get the Stokes intensity (W/m$^2$). |
| `.getIL()` |  | 2D float | Get the laser intensity (W/m$^2$). |
| `.getIA()` |  | 2D float | Get the anti-Stokes intensity (W/m$^2$). |
| `.getAS()` |  | 2D complex | Get the Stokes envelope field (V/m$^2$). |
| `.getAL()` |  | 2D complex | Get the laser envelope field (V/m$^2$). |
| `.getAA()` |  | 2D complex | Get the anti-Stokes envelope field (V/m$^2$). |
| `.setAS(A)` | 2D complex | | Set the Stokes envelope to a field `A` (V/m$^2$). |
| `.setAL(A)` | 2D complex | | Set the laser envelope to a field `A` (V/m$^2$). |
| `.setAA(A)` | 2D complex | | Set the anti-Stokes envelope to a field `A` (V/m$^2$). |
| `.getEnField(A,n)` | 2D complex, float | float | Get the energy (J) of an envelope field `A` (V/m) with index of refraction `n`. |
