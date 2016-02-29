import pandas as pd
import numpy as np
from uncertainties import unumpy
from uncertainties import ufloat

''' Functions to calculate various structural parameters '''

# Function to calculate gas mass. Takes as inputs: distance (Mpc), HI flux (Jy km/s),
# CO flux (Jy km/s) and CO flux error (Jy km/s). Returns: HI mass and error (assuming
# 20% flux measurement uncertainty), H2 mass and error (from inputs)
# and total gas mass and propagated error.
def calcGasMass(DMpc, FHI, FCO, FCOerr):
	# Calculate HI gas mass
	FHI = unumpy.uarray(FHI,(FHI*0.2))
	MHI = (2.356E5)*((DMpc)**2.)*(FHI)
	# Calculate H2 gas mass
	FCO = unumpy.uarray(FCO,FCOerr)
	MH2 = 7845*FCO*((DMpc)**2.)
	where_are_nans = unumpy.isnan(MH2)
	MH2[where_are_nans] = 0
	# Total gas mass
	Mgas = MHI + MH2
	return MHI, MH2, Mgas

# Function to calculate stellar mass. Takes as inputs: distance (Mpc), total
# integrated 3.6 micron apparent magnitude (extrapolated, not R25) and 
# mass-to-light ratio. Returns: total 3.6 micron stellar luminosity, stellar mass
# and error on stellar mass assuming +/- 0.1 M/L uncertainty.
def calcStellarMass(DMpc, mNIR, MLrat):
	Dpc = DMpc*(1E6)
	M_NIR = mNIR - 5.*np.log10(Dpc) + 5.
	L_star = (10**((3.24-M_NIR)/2.5))
	M_star = L_star * MLrat
	M_star_err = (L_star*(MLrat+0.1)) - M_star
	M_star = unumpy.uarray(M_star,M_star_err)
	return L_star, M_star

# Function to calculate extinction corrected magnitudes. Takes as inputs: observed
# integrated magnitude, error on that magnitude and extinction correction.
# Returns: extinction-corrected magnitude with error.
def extinctMag(m, merr, A):
	m = unumpy.uarray(m,merr)
	m0 = m - A
	return m0

# Function to calculate absolute magnitudes. Takes as inputs: distance (Mpc),
# total extinction-corrected integrated magnitude and error on that magnitude.
# Returns: extinction-corrected absolute magnitude with error.
def absMag(DMpc, mtot0, merr):
	Dpc = DMpc*(1E6)
	M = mtot0-5.*np.log10(Dpc)+5.
	M = unumpy.uarray(M,merr)
	return M
	
# Function to calculate SFR. Takes as inputs: distance (Mpc), log of the Halpha
# flux and error on that flux measurement. Returns: star formation rate with error.
def calcSFR(DMpc, logFHa, logFHa_err):
	Dpc = DMpc*(1E6)
	logFHa_arr = unumpy.uarray(logFHa,logFHa_err)
	F_Ha = 10**(logFHa_arr)
	L_Ha = F_Ha*(4*np.pi*((Dpc*(3.086E18))**2.))
	logsfr = unumpy.log10(L_Ha) - 41.27
	sfr = 10**(logsfr)
	return sfr

# Function to transform angular value to linear distance-corrected value. Takes
# as inputs: distance (Mpc) and angular size (arcseconds). Returns: linear
# value (kpc).
def linearize(DMpc, x):
	#x = unumpy.uarray(x,xerr)
	xkpc = (x/206.265)*(DMpc)
	return xkpc