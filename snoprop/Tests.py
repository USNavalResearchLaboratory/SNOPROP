# The source code has been authored by federal and non-federal employees. To the extent that a federal employee is an author of a portion of this software or a derivative work thereof, no copyright is claimed by the United States Government, as represented by the Secretary of the Navy ("GOVERNMENT") under Title 17, U.S. Code. All Other Rights Reserved. To the extent that a non-federal employee is an author of a portion of this software or a derivative work thereof, the work was funded in whole or in part by the U.S. Government, and is, therefore, subject to the following license: The Government has unlimited rights to use, modify, reproduce, release, perform, display, or disclose the computer software and computer software documentation in whole or in part, in any manner and for any purpose whatsoever, and to have or authorize others to do so. Any other rights are reserved by the copyright owner.

# Neither the name of NRL or its contributors, nor any entity of the United States Government may be used to endorse or promote products derived from this software, nor does the inclusion of the NRL written and developed software directly or indirectly suggest NRL's or the United States Government's endorsement of this product. 

# THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

'''
Test is the class for every test written for this code. Each test will create a "Test" object by importing this class and define a test function which will run some kind of simulation. The test function should return a 2-tuple with 1: true or false (whether the test was passed) and 2: a message containing more details about the test results. The function must also accept as a parameter a dict of constants which are to be defined in constants.py in this directory. This way, the tests can be easily re-run for any set of constants.
'''

# Class for each test to be implemented.
class Test:
    def __init__(self, name, function):
        self.name, self.function = name, function
    def run(self, constants):
        return self.function(constants)

# Class to which all tests are added and then run together. See usage in run_tests.py
class Tests:
    tests = []

    def add(self, test):
        self.tests.append(test)

    def run(self, constants, verbose=False):
        print("Running {:d} tests".format(len(self.tests)))
        print("Using the following constants:")
        for key,val in constants.items():
            print(str(key)+": "+str(val))
        results = []
        for i,test in enumerate(self.tests):
            print("{:02d}:         Running test {}".format(i,test.name))
            result, message = test.run(constants)
            if verbose:
                print(message)
            results.append(result)
            resText = "Success" if result else "Failure"
            print("{:02d}: {}: {}".format(i,resText,test.name))
        totResult = all(results)
        resText = "Success" if result else "Failure"
        print("Finished testing")
        print("{}: {:d} passed and {:d} failed.".format(resText,sum(results),len(self.tests) - sum(results)))


def run_tests(v=False):
    import time

    from .test_constants import constants

    tests = Tests()

    from . import test_ionization
    tests.add(test_ionization.test)

    from . import test_four_wave_mixing
    tests.add(test_four_wave_mixing.test)

    from . import test_raman
    tests.add(test_raman.test)

    from . import test_plasma_focusing
    tests.add(test_plasma_focusing.test)

    from . import test_diffraction
    tests.add(test_diffraction.test)

    from . import test_linear_focusing
    tests.add(test_linear_focusing.test)

    from . import test_pcrit
    tests.add(test_pcrit.test)
 

    t0 = time.time()
    tests.run(constants, verbose=v)
    t1 = time.time()
    print("{:.3f} seconds elapsed".format(t1-t0))
