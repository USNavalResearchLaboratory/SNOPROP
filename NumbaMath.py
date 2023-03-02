# The source code has been authored by federal and non-federal employees. To the extent that a federal employee is an author of a portion of this software or a derivative work thereof, no copyright is claimed by the United States Government, as represented by the Secretary of the Navy ("GOVERNMENT") under Title 17, U.S. Code. All Other Rights Reserved. To the extent that a non-federal employee is an author of a portion of this software or a derivative work thereof, the work was funded in whole or in part by the U.S. Government, and is, therefore, subject to the following license: The Government has unlimited rights to use, modify, reproduce, release, perform, display, or disclose the computer software and computer software documentation in whole or in part, in any manner and for any purpose whatsoever, and to have or authorize others to do so. Any other rights are reserved by the copyright owner.

# Neither the name of NRL or its contributors, nor any entity of the United States Government may be used to endorse or promote products derived from this software, nor does the inclusion of the NRL written and developed software directly or indirectly suggest NRL's or the United States Government's endorsement of this product. 

# THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

import numpy as np
import numba

PI = np.pi

@numba.njit(fastmath=True)
def dNedtau(WMPI_NH2O,vi,ne,eta):
    return WMPI_NH2O + vi*ne - eta*ne


@numba.njit(fastmath=True)
def getI(A,factor):
    return np.real(A*np.conj(A))*(factor*4.)

@numba.njit(fastmath=True)
def easyPow(x,n): # Raises x to the integer power x. Only for small n.
    res = x
    for i in range(n-1):
        res *= x
    return res

@numba.njit(fastmath=True)
def factorial(n):
    res = 1
    for i in range(2,n+1):
        res *= i
    return res

#---------------------------------------------- W_ofi(E,ω) --------------------------------------------
# This subroutine calculates the Keldysh rate by summing up the MPI and tunnel rates. This is not the
# original Keldysh formula, but rather, a shortcut formul for computational efficincy (se tech notes).
# W_mpi=4ω/(9π)*(mω/ħ)^(3/2)*F[(2N-2x)^(1/2)]*exp(2*N)*(16γ^2+8)^(-N),     for γ>>1 -> Balling, Eq. 8.
# W_tun=4/(9π^2)*(∆/ħ)*(m*∆/ħ^2)^(3/2)*[ħω/(γ*∆)]^(5/2)*exp[-π*∆*γ/(2ħω)], for γ<<1 -> Balling, Eq. 9.
# where N=int(x+1) is the number of photons with x=(∆+Up)/(ħω). F(x)=exp(-x^2)*Integral_0^x{exp(y^2)*dy}
# is the Dawson integral. The Keldysh parameter is γ=(m*∆)^(1/2)*ω/(e*E).
# input:
#   - E         : laser field E in [V/m]
#   - omega     : laser frequency ω=2πc/λ in [rad/s]
#   - Ip        : ionization potential in units [eV] (parameter ∆ in formulas)
#   reduced_mass: reduced electron mass in units of electron mass, e.g 0.5
# output:
#   - W_ofi     : ionization rate in units [m^-3*s^-1]
#
# The original formulas of Keldysh have been corrected following Balling and Gruzdev.
# P. Balling, Rep. Prog. Phys. 76, 036502 (2013)
# V. E. Gruzdev, Laser-Induced Damage In Optical Materials: Proceedings of the Society of Photo-Optical
# Instrumentation Engineers (Proc. SPIE), vol. 5647 (2004) 
#
@numba.njit(fastmath=True)
def Get_ofi(E,omega,Ip,reduced_mass):                              # begin subroutine
    # Dawson integral F(x)=exp(-x^2)*Integral_0^x{exp(y^2)*dy} 
    def F(x): return x*(1.0+0.3661869354*x*x)/(1.0+0.8212009121*x*x+2.0*0.3661869354*x**4)
    # atomic physics constants
    pi=3.14159                                                     # π
    e0=1.6022e-19                                                  # electron charge in [C]
    me=9.1094e-31                                                  # electron mass in [kg]
    c=3.0e8                                                        # speed of light in [m/s]
    h_bar=1.0545e-34                                               # Plank's constant ħ in [J*s]
    # calculate MPI ionization rate
    m=me*reduced_mass                                              # convert reduced to actual mass in [kg]
    Ip*=e0                                                         # ionization threshold Ip in [J]
    gama=np.sqrt(m*Ip)*omega/(e0*E)                                # Keldysh parameter γ
    Up=(e0*E)**2/(4.0*m*omega**2)                                  # ponderomotive energy Up in [J]
    x=(Ip+Up)/(h_bar*omega)                                        # x=(I+Up)/(ħω)
    x=min(x,200.0)                                                 # limit x
    N=int(x+1.0)                                                   # order of MPI
    y=2.0*(N-x)                                                    # y=2*(<x+1>-x)
    if N*np.log(gama) > 100: return 1.0e-90;                       # return to prevent very small numbers
    W_mpi=4.0*omega/(9.0*pi)*(m*omega/h_bar)**1.5*F(np.sqrt(y))*np.exp(2.0*N)/(16.0*gama**2+8.0)**N
    if gama > 10.0: return W_mpi                                   # return W_mpi if γ > 10
    # calculate tunnel ionization rate
    y=h_bar*omega/(gama*Ip)                                        # y=ħω/(γ*Ip)
    W_tun=4.0*Ip/(9.0*pi**2*h_bar)*(m*Ip/h_bar**2)**1.5*y**2.5*np.exp(-0.5*pi/y)
    if gama < 0.1: return W_tun                                    # return W_tun if γ < 0.1
    # calculate total OFI rate
    return W_mpi+W_tun                                             # return W in [m^-3*s^-1]
#------------------------------------------------- end -------------------------------------------------
#----------------------------------------------- Te(E,ω) -----------------------------------------------
# This subroutine solves the power balance equation for electrons to compute the electron temperature.
# input:
#   - E  : laser field in [V/m]
#   - ω  : laser frequency in [rad/s]
#   - Te : electron temperature in [eV]; optional
# output:
#   - Te : electron temperature in [eV]
#
# The electron temperature is advanced according to the power balance equation:
#   d/dt(1.5*ne*Te)=P_joule-(P_mom+P_vib1+P_vib2+P_vib3+P_att+P_exc+P_ion)
# The power deposition rates are:
#   P_joule=(e*E)^2*ne*ν_mom/[2*me*(ν_mom^2+ω^2)] in [W/m^3]
#   P_mom  =(3*me/M)*ne*ν_mom
#   P_vib1 =ne*ν_vib1*∆E_vib1, where ∆E_vib1=0.20 eV
#   P_vib2 =ne*ν_vib2*∆E_vib2, where ∆E_vib2=0.48 eV
#   P_vib3 =ne*ν_vib3*∆E_vib3, where ∆E_vib3=1.00 eV
#   P_att  =ne*ν_att*∆E_att,   where ∆E_att=4.50 eV
#   P_exc  =ne*ν_exc*∆E_exc,   where ∆E_exc=8.70 eV
#   P_ion  =ne*ν_ion*∆E_ion,   where ∆E_ion=13.1 eV
#
# The Joule heating terms is, by definition, P_joule=(e*E)^2*ne*ν_mom/[2*me*(ν_mom^2+ω^2)] in [W/m^3].
# The electron density on both sides has been canceled out (otherwise it has to be an input parameter).
# In addition, all power balance terms are calculated in units [eV/s] rather than in [W] in order to
# avoid the conversion from [eV] to [K]. All collisional rates, i.e. nu_mom, nu_vib1, etc. are in units
# [1/s], in which Te is in units of [eV]. The rates are calculated by integrating the coresponding cross
# section over Maxwellian electron energy istribution. The cross sections are from Ref. [1]
# [1] H. Date, K. L. Sutherland, H. Hasegawa, M. Shimozuma, "Ionization and excitation collision processes 
# of electrons in liquid water", Nuclear Instr. Methods in Physics Research B 265, 515-520 (2007)
#
@numba.njit(fastmath=True)
def Get_Te_steadystate(ES,wS,EL,wL,EA,wA,Te=1.0):
    # ES,EL and EA -> laser fields in [V/m], wS, wL and wA -> laser frequency in [rad/s]
    # atomic physics constants
    e0=1.6022e-19                                                            # electron charge in [C]
    me=9.1094e-31                                                            # electron mass in [kg]
    # computational parameters
    Nmb_iter=30                                                              # max number of iterations
    eps=1.0e-3                                                               # convergence criteria
    Te=max(Te,0.1)                                                           # make Te larger than room temp
    for iter in range(Nmb_iter):                                             # begin loop to compute Te
        # calculate collision rates in units [1/s] and power gain or loss in units [eV/s]
        nu_mom =3.4e+15*Te**0.52/(1+0.01*Te**1.5);P_mom =nu_mom*9.08e-5*Te   # momentum transfer
        nu_vib1=1.017e+14*Te**-0.1436*np.exp(-0.1344/Te);P_vib1=nu_vib1*0.20 # vib. excitation #1
        nu_vib2=2.713e+14*Te**-0.3962*np.exp(-0.4361/Te);P_vib2=nu_vib2*0.48 # vib. excitation #2
        nu_vib3=7.107e+13*Te** 0.0784*np.exp(-0.9280/Te);P_vib3=nu_vib3*1.00 # vib. excitation #3
        nu_att =2.778e+13*Te**-0.7231*np.exp(-5.2515/Te);P_att =nu_att*4.50  # attachment
        nu_exc =1.249e+13*Te**1.20480*np.exp(-8.9390/Te);P_exc =nu_exc*8.70  # excitation
        nu_ion =4.720e+12*Te**1.84040*np.exp(-13.029/Te);P_ion =nu_ion*13.1  # ionization
        # Joule heating
        P_joule=e0*nu_mom/(2*me)*(ES**2/(nu_mom**2+wS**2)+EL**2/(nu_mom**2+wL**2)+EA**2/(nu_mom**2+wA**2))
        # update Te
        power_balance=(P_mom+P_vib1+P_vib2+P_vib3+P_att+P_exc+P_ion)/P_joule # power loss/power gain
        Te/=power_balance**0.25                                              # advance Te
        if abs(1.0-power_balance) < eps: break                               # exit if convergence reached
    Te=max(Te,0.026)                                                         # make Te larger than room temp
    return [Te,nu_ion,nu_att]                                                # return Te
#------------------------------------------------- end -------------------------------------------------

@numba.njit(parallel=True, fastmath=True)
def calcNeMPI(fieldsIn, wmpiOut, ve, ne, te, constants):                     # comment GMP: added array te
    AS, AL, AA = fieldsIn
    WMPI_NH2O, WMPI_NH2O_S, WMPI_NH2O_L,WMPI_NH2O_A = wmpiOut
    dt, eta, wS, wL, wA, nS, nL, nA, e0, c, NH2O, lS,lL,lA, IMPI, veC, viC = constants

    tlen, rlen = AS.shape

    ISfactor = (nS/2*e0*c)
    ILfactor = (nL/2*e0*c)
    IAfactor = (nA/2*e0*c)
    wfacS= NH2O*2*np.pi/factorial(lS-1)
    wfacL= NH2O*2*np.pi/factorial(lL-1)
    wfacA= NH2O*2*np.pi/factorial(lA-1)

    Te=1.0                                                                   # initialize Te
    for i in numba.prange(rlen-1): # Solve for electron density using RK4 and Eq. 10 from Hafizi 2016

        ne[0,i] = 0. # Boundary condition is that we have no electon density before the pulse arrives

        # assign fields to local variables
        ASa = np.absolute(AS[0,i])/2.0
        ALa = np.absolute(AL[0,i])/2.0
        AAa = np.absolute(AA[0,i])/2.0

        # WMPI (original version)
        #WMPI_NH2O_S[0,i] = wfacS*wS*easyPow(getI(AS[0,i],ISfactor)/IMPI, lS)
        #WMPI_NH2O_L[0,i] = wfacL*wL*easyPow(getI(AL[0,i],ILfactor)/IMPI, lL)
        #WMPI_NH2O_A[0,i] = wfacA*wA*easyPow(getI(AA[0,i],IAfactor)/IMPI, lA)
        #WMPI_NH2O[0,i] = WMPI_NH2O_S[0,i] + WMPI_NH2O_L[0,i] + WMPI_NH2O_A[0,i]

        # <ve>=e0*|E|/(me*ω)
        fieldsAvg = np.sqrt(ASa*ASa/(wS*wS) + ALa*ALa/(wL*wL) + AAa*AAa/(wA*wA))
        ve[0,i] = veC*fieldsAvg

        # collisional ionization rate (original version)
        #vinext = viC*ve[0,i]*fieldsAvg*fieldsAvg

        # calculate Keldysh OFI rates with Ip=9.5 eV and electron reduced mass 0.2 (comment GMP: use Keldysh rate by GMP)
        WMPI_NH2O_S[0,i] = Get_ofi(ASa,wS,9.5,0.2)
        WMPI_NH2O_L[0,i] = Get_ofi(ALa,wL,9.5,0.2)
        WMPI_NH2O_A[0,i] = Get_ofi(AAa,wA,9.5,0.2)
        WMPI_NH2O[0,i] = WMPI_NH2O_S[0,i] + WMPI_NH2O_L[0,i] + WMPI_NH2O_A[0,i]

        # calculate Te and collisional rates (comment GMP: use model developed by GMP)
        Te=te[0,i]
        Te,nu_ion,nu_att=Get_Te_steadystate(ASa,wS,ALa,wL,AAa,wA,Te)
        te[0,i]=Te      # comment GMP: use model developed by GMP
        vinext = nu_ion # comment GMP: use model developed by GMP
        eta=nu_att      # comment GMP: use model developed by GMP

        if tlen==1:
            ne[0,i] = WMPI_NH2O[0,i]/(eta - vinext)

        for j in range(tlen-1):

            # assign fields to local variables
            ASa = np.absolute(AS[j+1,i])
            ALa = np.absolute(AL[j+1,i])
            AAa = np.absolute(AA[j+1,i])

            # WMPI calculation (original version)
            #WMPI_NH2O_S[j+1,i] = wfacS*wS*easyPow(getI(AS[j+1,i],ISfactor)/IMPI, lS)
            #WMPI_NH2O_L[j+1,i] = wfacL*wL*easyPow(getI(AL[j+1,i],ILfactor)/IMPI, lL)
            #WMPI_NH2O_A[j+1,i] = wfacA*wA*easyPow(getI(AA[j+1,i],IAfactor)/IMPI, lA)
            #WMPI_NH2O[j+1,i] = WMPI_NH2O_S[j+1,i] + WMPI_NH2O_L[j+1,i] + WMPI_NH2O_A[j+1,i]

            # <ve>=e0*|E|/(me*ω)
            fieldsAvg = np.sqrt(ASa*ASa/(wS*wS) + ALa*ALa/(wL*wL) + AAa*AAa/(wA*wA))
            ve[j+1,i] = veC*fieldsAvg # We save ve in an array for later

            # collisional ionization rate (original version)
            #vihere = vinext # This is from the previous loop
            #vinext = viC*ve[j+1,i]*fieldsAvg*fieldsAvg

            # calculate OFI rates with Ip=9.5 eV and electron reduced mass 0.2 (comment GMP: use Keldysh rate by GMP)
            WMPI_NH2O_S[j+1,i] = Get_ofi(ASa,wS,9.5,0.2)
            WMPI_NH2O_L[j+1,i] = Get_ofi(ALa,wL,9.5,0.2)
            WMPI_NH2O_A[j+1,i] = Get_ofi(AAa,wA,9.5,0.2)
            WMPI_NH2O[j+1,i] = WMPI_NH2O_S[j+1,i] + WMPI_NH2O_L[j+1,i] + WMPI_NH2O_A[j+1,i]

            # calculate Te and collisional rates (comment GMP: use model developed by GMP)
            Te=te[j+1,i]
            Te,nu_ion,nu_att=Get_Te_steadystate(ASa,wS,ALa,wL,AAa,wA,Te)
            te[j+1,i]=Te
            vihere = vinext # This is from the previous loop
            vinext = nu_ion # comment GMP: use model developed by GMP
            eta = nu_att    # comment GMP: use model developed by GMP

            vihalf = (vihere+vinext)/2.
            
            #Ne calculation
            Nehere = ne[j,i]
            WMPI_NH2Ohere = WMPI_NH2O[j,i]
            WMPI_NH2Onext = WMPI_NH2O[j+1,i]
            WMPI_NH2Ohalf = (WMPI_NH2Ohere+WMPI_NH2Onext)/2.
            # Now use regular RK4 integration
            k1 = dt*(WMPI_NH2Ohere + vihere*Nehere - eta*Nehere)
            ne1 = Nehere + k1/2.
            k2 = dt*(WMPI_NH2Ohalf + vihalf*ne1 - eta*ne1)
            ne1 = Nehere + k2/2.
            k3 = dt*(WMPI_NH2Ohalf + vihalf*ne1 - eta*ne1)
            ne1 = Nehere + k3
            k4 = dt*(WMPI_NH2Onext + vinext*ne1 - eta*ne1)
            ne[j+1,i] = Nehere + 1/6.*(k1 + 2.*k2 + 2.*k3 + k4)
            

@numba.njit(fastmath=True)
def rDeriv(arr, ti, ri, rlen, drd): # Get the first r derivative of arr at indices (ti,ri). drd should equal dr*2
    # # Fourth-order solver
    # diff = 0.
    # if ri == rlen-1:
    #     diff = -25/12*arr[ti,ri]+4*arr[ti,ri-1]-3*arr[ti,ri-2]+4/3*arr[ti,ri-3]-1/4*arr[ti,ri-4]
    #     diff *= -1
    # elif ri == rlen-2:
    #     diff = (-3*arr[ti,ri+1]-10*arr[ti,ri]+18*arr[ti,ri-1]-6*arr[ti,ri-2]+1*arr[ti,ri-3])/12
    #     diff *= -1
    # elif ri == 0:
    #     diff = -25/12*arr[ti,ri]+4*arr[ti,ri+1]-3*arr[ti,ri+2]+4/3*arr[ti,ri+3]-1/4*arr[ti,ri+4]
    # elif ri == 1: 
    #     diff = (-3*arr[ti,ri-1]-10*arr[ti,ri]+18*arr[ti,ri+1]-6*arr[ti,ri+2]+1*arr[ti,ri+3])/12
    # else:
    #     diff = 1/12*arr[ti,ri-2] -2/3*arr[ti,ri-1] + 2/3*arr[ti,ri+1] - 1/12*arr[ti,ri+2]
    # return diff/(drd/2) # drd is 2*dr

    # Second-order solver
    diff = 0.
    if ri == rlen-1:
        diff = 3.*arr[ti,ri]-4.*arr[ti,ri-1]+arr[ti,ri-2]
    elif ri == 0:
        diff = -3.*arr[ti,0]+4.*arr[ti,1]-arr[ti,2]
    else:
        diff = arr[ti,ri+1] - arr[ti,ri-1]
    return diff/(drd) # drd is 2*dr

@numba.njit(fastmath=True)
def r2Deriv(arr, ti, ri, rlen, dr2): # Get the second r derivative of arr at indices (ti,ri). dr2 should equal dr**2
    # # Fourth-order solver
    # if ri == rlen-1:
    #     diff = 15/4*arr[ti,ri]-77/6*arr[ti,ri-1]+107/6*arr[ti,ri-2]-13*arr[ti,ri-3] + 61/12*arr[ti,ri-4] - 5/6*arr[ti,ri-5]
    # elif ri == rlen-2:
    #     diff = (10*arr[ti,ri+1]-15*arr[ti,ri]-4*arr[ti,ri-1]+14*arr[ti,ri-2]-6*arr[ti,ri-3]+1*arr[ti,ri-4])/12
    # elif ri == 0: 
    #    #diff = 2*arr[ti,ri]-5*arr[ti,ri+1]+4*arr[ti,ri+2]-arr[ti,ri+3]
    #     #diff = 35/12*arr[ti,ri]-26/3*arr[ti,ri+1]+19/2*arr[ti,ri+2]-14/3*arr[ti,ri+3]+11/12*arr[ti,ri+4]
    #     diff = 15/4*arr[ti,ri]-77/6*arr[ti,ri+1]+107/6*arr[ti,ri+2]-13*arr[ti,ri+3] + 61/12*arr[ti,ri+4] - 5/6*arr[ti,ri+5]
    # elif ri == 1:
    #     diff = (10*arr[ti,ri-1]-15*arr[ti,ri]-4*arr[ti,ri+1]+14*arr[ti,ri+2]-6*arr[ti,ri+3]+1*arr[ti,ri+4])/12
    # else:
    #     diff = -1/12*arr[ti,ri-2] + 4/3*arr[ti,ri-1] -5/2*arr[ti,ri] + 4/3*arr[ti,ri+1] -1/12*arr[ti,ri+2]
    # return diff/dr2

    # Second-order solver
    diff = 0.
    if ri == rlen-1:
        diff = 2.*arr[ti,ri]-5.*arr[ti,ri-1]+4.*arr[ti,ri-2]-arr[ti,ri-3]
    elif ri == 0:
        diff = 2*arr[ti,0]-5*arr[ti,1]+4*arr[ti,2]-arr[ti,3]
    else:
        diff = arr[ti,ri+1] + arr[ti,ri-1] - 2.*arr[ti,ri]
    return diff/(dr2) # dr2 is dr**2

@numba.njit(fastmath=True)
def tDeriv(arr, ti, ri, tlen, dtd): # Get the first t derivative of arr at indices (ti,ri). dtd should equal dt*2
    diff = 0.
    if ti == tlen-1:
        diff = 3.*arr[ti,ri]-4.*arr[ti-1,ri]+arr[ti-2,ri]
    elif ti == 0:
        diff = -3.*arr[0,ri]+4.*arr[1,ri]-arr[2,ri]
    else:
        diff = arr[ti+1,ri] - arr[ti-1,ri]
    return diff/(dtd) # dtd is 2*dt

@numba.njit(fastmath=True)
def t2Deriv(arr, ti, ri, tlen, dt2): # Get the second r derivative of arr at indices (ti,ri). dr2 should equal dr**2
    diff = 0.
    if ti == tlen-1:
        diff = 2.*arr[ti,ri]-5.*arr[ti-1,ri]+4.*arr[ti-2,ri]-arr[ti-3,ri]
    elif ti == 0:
        diff = 2*arr[0,ri]-5*arr[1,ri]+4*arr[2,ri]-arr[3,ri]
    else:
        diff = arr[ti+1,ri] + arr[ti-1,ri] - 2.*arr[ti,ri]
    return diff/(dt2) # dt2 is dt**2

def getRHS(constants, includes, euler):
    cS, cL, cA, dd = constants
    cS1, cS2, cS3, cS4, cS5, cS6, cS7, cS8, cS9, cS10 = cS
    cL1, cL2, cL3, cL4, cL5, cL6, cL7, cL8, cL9, cL10 = cL
    cA1, cA2, cA3, cA4, cA5, cA6, cA7, cA8, cA9, cA10 = cA

    dt, dr, tlen, rlen = dd
    drd = 2.*dr
    dtd = 2.*dt
    dr2 = dr*dr
    dt2 = dt*dt

    include_stokes, include_antistokes, include_ionization, include_plasma_refraction, include_energy_loss, updateS, updateL, updateA = includes
    #print('getrhs',constants,includes)
    #print(cL6,cL7,cL8)
    
    @numba.njit(parallel=True, fastmath=True)
    def RHS(ASorig,ALorig,AAorig, # fields from previous zstep
            AS,AL,AA, # fields used to calculate new dAdz (from the previous zstep for euler=True, from the last iteration of current zstep if euler=False)
            dASdz, dALdz, dAAdz, # Previous estimates for dAdz (either from Euler or previous Crank-Nicolson iteration). Not used if euler=True.
            ASout, ALout, AAout, # Save the new fields
            dASdzOut, dALdzOut, dAAdzOut, # Save the dAdz if euler = True
            WMPI_NH2O_S, WMPI_NH2O_L, WMPI_NH2O_A, # MPI rates
            Ne, Ve, eikz, dz): # Electron density, collision frequency, exp(i dk z) factor, and z step size

        errorArr = np.zeros(rlen) # Store the error row-by-row to avoid synchronization

        for ri in numba.prange(rlen): # numba.prange is parallel across threads
            rowErr = 0. # Keep track of the error for each row since we're doing this in parallel
            rinv = 0.0 # r inverse (set to 0 at r=0 so it isn't infinity)
            if ri > 0:
                rinv = 1/(ri*dr)#+dr/2)
            for ti in numba.prange(tlen):



                # Get local values of the fields, values squared, and derivatives
                if include_stokes:
                    AS1 = AS[ti,ri]
                    ASa2 = AS1*np.conj(AS1)
                    dASdr = rDeriv(AS, ti, ri, rlen, drd)
                    asin = AS[ti,ri] # Grab these values before we write in case AS = ASout
                else:
                    AS1, ASa2 = 0,0
                    
                AL1 = AL[ti,ri]
                ALa2 = AL1*np.conj(AL1)
                dALdr = rDeriv(AL, ti, ri, rlen, drd)
                alin = AL[ti,ri]
                if include_antistokes:
                    AA1 = AA[ti,ri]
                    AAa2 = AA1*np.conj(AA1)
                    dAAdr = rDeriv(AA, ti, ri, rlen, drd)
                    aain = AA[ti,ri]
                else:
                    AA1, AAa2 = 0,0

                if include_ionization or include_plasma_refraction or include_energy_loss:
                    ne = Ne[ti,ri]
                    ve = Ve[ti,ri]


                # Initialize solutions
                Sres = 0
                Lres = 0
                Ares = 0

                # Now we will use
                if include_stokes and updateS:
                    Sres += cS1*(r2Deriv(AS, ti, ri, rlen, dr2) + rinv*dASdr) # Diffraction and linear focusing
                    if tlen>1:
                        if cS2 != 0: Sres += cS2*tDeriv(AS, ti, ri, tlen, dtd) # Group delay
                        if cS10 != 0: Sres += cS10 * t2Deriv(AS,ti,ri,tlen,dt2) # Group velocity dispersion
                    if cS3 != 0: Sres += cS3*ASa2*AS1 # self-phase modulation
                    if cS4 != 0: Sres += cS4*ALa2*AS1 # SRS and cross-phase modulation
                    if cS5 != 0: Sres += cS5*AAa2*AS1 # cross-phase modulation
                    if cS6 != 0: Sres += cS6*AL1*AL1*np.conj(AA1)*eikz # FWM and cross-phase modulation
                    if cS7 != 0: Sres += cS7*ne*AS1 # Plasma refraction
                    if cS8 != 0: Sres += cS8*ne*ve*AS1 # Plasma energy loss
                    if cS9 != 0: Sres += cS9*WMPI_NH2O_S[ti,ri]*AS1/ASa2 # Plasma energy loss

                if updateL:
                    Lres += cL1*(r2Deriv(AL, ti, ri, rlen, dr2) + rinv*dALdr)
                    if cL3 != 0: Lres += cL3*ASa2*AL1
                    if cL4 != 0: Lres += cL4*ALa2*AL1
                    if cL5 != 0: Lres += cL5*AAa2*AL1
                    if cL6 != 0: Lres += cL6*np.conj(AL1)*AS1*AA1*np.conj(eikz)
                    if cL7 != 0: Lres += cL7*ne*AL1
                    if cL8 != 0: Lres += cL8*ne*ve*AL1
                    if cL9 != 0: Lres += cL9*WMPI_NH2O_L[ti,ri]*AL1/ALa2
                    if tlen>1:
                        if cL10 != 0: Lres += cL10 * t2Deriv(AL,ti,ri,tlen,dt2)

                if include_antistokes and updateA:
                    Ares += cA1*(r2Deriv(AA, ti, ri, rlen, dr2) + rinv*dAAdr) 
                    if tlen>1:
                        if cA2 != 0: Ares += cA2*tDeriv(AA, ti, ri, tlen, dtd)
                        if cA10 != 0: Ares += cA10 * t2Deriv(AA,ti,ri,tlen,dt2)
                    if cA3 != 0: Ares += cA3*ASa2*AA1
                    if cA4 != 0: Ares += cA4*ALa2*AA1
                    if cA5 != 0: Ares += cA5*AAa2*AA1
                    if cA6 != 0: Ares += cA6*AL1*AL1*np.conj(AS1)*eikz
                    if cA7 != 0: Ares += cA7*ne*AA1
                    if cA8 != 0: Ares += cA8*ne*ve*AA1
                    if cA9 != 0: Ares += cA9*WMPI_NH2O_A[ti,ri]*AA1/AAa2

                    

                # Now use these dAdz values to update the fields!
                if euler: # Use Euler method
                    if include_stokes:
                        ASout[ti,ri] = ASorig[ti,ri] + dz*Sres
                        dASdzOut[ti,ri] = Sres # Save dAdz for future use
                    ALout[ti,ri] = ALorig[ti,ri] + dz*Lres
                    dALdzOut[ti,ri] = Lres
                    if include_antistokes:
                        AAout[ti,ri] = AAorig[ti,ri] + dz*Ares
                        dAAdzOut[ti,ri] = Ares
                else: # Crank-Nicolson
                    if include_stokes:
                        ASout[ti,ri] = ASorig[ti,ri] + dz * (Sres + dASdz[ti,ri])/2.0
                        rowErr += abs(asin - ASout[ti,ri]) # save the error between current and previous iterations
                    ALout[ti,ri] = ALorig[ti,ri] + dz * (Lres + dALdz[ti,ri])/2.0
                    rowErr += abs(alin - ALout[ti,ri])
                    if include_antistokes:
                        AAout[ti,ri] = AAorig[ti,ri] + dz * (Ares + dAAdz[ti,ri])/2.0
                        rowErr += abs(aain - AAout[ti,ri])
                        

            errorArr[ri] = rowErr

        if euler: # We don't return error for Euler method
            return
        else: # Return error for Crank-Nicolson method
            return np.sum(errorArr) / float(rlen*tlen) # Return the error per cell

    return RHS

