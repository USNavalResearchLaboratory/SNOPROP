# The source code has been authored by federal and non-federal employees. To the extent that a federal employee is an author of a portion of this software or a derivative work thereof, no copyright is claimed by the United States Government, as represented by the Secretary of the Navy ("GOVERNMENT") under Title 17, U.S. Code. All Other Rights Reserved. To the extent that a non-federal employee is an author of a portion of this software or a derivative work thereof, the work was funded in whole or in part by the U.S. Government, and is, therefore, subject to the following license: The Government has unlimited rights to use, modify, reproduce, release, perform, display, or disclose the computer software and computer software documentation in whole or in part, in any manner and for any purpose whatsoever, and to have or authorize others to do so. Any other rights are reserved by the copyright owner.

# Neither the name of NRL or its contributors, nor any entity of the United States Government may be used to endorse or promote products derived from this software, nor does the inclusion of the NRL written and developed software directly or indirectly suggest NRL's or the United States Government's endorsement of this product. 

# THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

# This script will run the tests in this directory (using the constants in constants.py) and print the status.

from Tests import Test,Tests
import time, sys

from constants import constants

if len(sys.argv) != 2:
    print('Usage: python run_test.py [Test Name]')
    quit()

tests = Tests()

import test_ionization
tests.add(test_ionization.test)

import test_four_wave_mixing
tests.add(test_four_wave_mixing.test)

import test_raman
tests.add(test_raman.test)

import test_plasma_focusing
tests.add(test_plasma_focusing.test)

import test_diffraction
tests.add(test_diffraction.test)

import test_linear_focusing
tests.add(test_linear_focusing.test)

import test_pcrit
tests.add(test_pcrit.test)

found=False
tests2 = Tests()
for t in tests.tests:
    if t.name == sys.argv[1]:
        found=True
        tests2.add(t)
if not found:
    print('Must specify valid test name. Must be one of ',[t.name for t in tests.tests])
    quit()

t0 = time.time()
tests2.run(constants, verbose=True)
t1 = time.time()
print("{:.3f} seconds elapsed".format(t1-t0))
