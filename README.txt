----------- SNOPROP ------------
SNOPROP: Solver for Nonlinear Optical Propagation is a Python-based simulation code.
Includes
	Linear laser propagation
	Nonlinear self-phase and cross-phase modulation
	Group velocity dispersion
	Multiphoton ionization
	Collisional ionization
	Stimulated Raman Scattering
	Four-wave mixing
	Laser pulse defined in time-radius coordinates and advanced in z.

All units are SI. Numbers are stored as doubles or complex doubles in the simulation where applicable. The output data (scalar data, 1d data, and 2d arrays) are saved as single-precision floats.


----------- License -------------
The source code has been authored by federal and non-federal employees. To the extent that a federal employee is an author of a portion of this software or a derivative work thereof, no copyright is claimed by the United States Government, as represented by the Secretary of the Navy ("GOVERNMENT") under Title 17, U.S. Code. All Other Rights Reserved. To the extent that a non-federal employee is an author of a portion of this software or a derivative work thereof, the work was funded in whole or in part by the U.S. Government, and is, therefore, subject to the following license: The Government has unlimited rights to use, modify, reproduce, release, perform, display, or disclose the computer software and computer software documentation in whole or in part, in any manner and for any purpose whatsoever, and to have or authorize others to do so. Any other rights are reserved by the copyright owner.

Neither the name of NRL or its contributors, nor any entity of the United States Government may be used to endorse or promote products derived from this software, nor does the inclusion of the NRL written and developed software directly or indirectly suggest NRL's or the United States Government's endorsement of this product. 

THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.


----------- Installation ----------
The program uses Numpy, Scipy, and Matplotlib extensively. In addition, the program uses Numba, a JIT compiler, to accelerate the math.
All of these are included by default with the Anaconda python distribution. This is the easiest way to use SNOPROP.
Requires Numba (version >= 0.42). You may want to double-check that Conda installed at minimum this version. You can type "conda list" to see the installed version of Numba.


------------ Testing ---------------
Tests are found in the tests/ folder. You can run them with
      python tests/run_scripts.py [-v]
where -v is optional for more verbose logging
This will run all existing tests and report the outcome. The constants used for the tests are saved in tests/constants.py


----------- Use ----------------
Create a python file. Import the Integrator class. Create a dictionary with the parameters listed below.
The shortest possible input deck might look something like this:
    from Integrator import Integrator
    params = {
        # Specify simulation parameters here
    }
    sim = Integrator(params)
    sim.run()
    
You can specify the number of threads to use with syntax like the following:
    NUMBA_NUM_THREADS=4 python simulate.py
You may also run it step-by-step with the sim.move() function
You can retrieve the simulation position with sim.getZ() and the field intensities with sim.getIS(), sim.getIL(), sim.getIA()
If you save restarts, you can just load the object from the restart file as so
    sim = pickle.load('restart_file','rb')
    sim.run()

Input parameters are detailed in the user guide.


