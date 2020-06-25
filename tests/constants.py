# The source code has been authored by federal and non-federal employees. To the extent that a federal employee is an author of a portion of this software or a derivative work thereof, no copyright is claimed by the United States Government, as represented by the Secretary of the Navy ("GOVERNMENT") under Title 17, U.S. Code. All Other Rights Reserved. To the extent that a non-federal employee is an author of a portion of this software or a derivative work thereof, the work was funded in whole or in part by the U.S. Government, and is, therefore, subject to the following license: The Government has unlimited rights to use, modify, reproduce, release, perform, display, or disclose the computer software and computer software documentation in whole or in part, in any manner and for any purpose whatsoever, and to have or authorize others to do so. Any other rights are reserved by the copyright owner.

# Neither the name of NRL or its contributors, nor any entity of the United States Government may be used to endorse or promote products derived from this software, nor does the inclusion of the NRL written and developed software directly or indirectly suggest NRL's or the United States Government's endorsement of this product. 

# THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

constants = {
    'wV': 2*3.14159*1.019e14, # The Raman line is about 3400/cm, which is 101.9 THz.
    'Uion': 9.5 * 1.602e-19, # in J
    'sigmaC': 1e-19, # collision cross section from Hafizi 2016 paper
    'IMPI': 1e18, # multiphoton ionization intensity From Hafizi 2016 paper
    'eta': 1.7e11, # Electron reattachment rate (plasma lifetime) in s^-1 (Hafizi 2016 paper)
    'IBackground': 1e2, # Background intensity in W/m^2
    'n2Kerr': 5.0e-20, # index of refraction n = n0 + n2*I. xNR will be calculated from this.
    'n2Raman': -1.7e-20j, # specify xRS/xNL. Should be imaginary.
    'wavelength': 355e-9,
}
