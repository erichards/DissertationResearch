import pandas as pd
import numpy as np
from uncertainties import unumpy
#from uncertainties import ufloat

''' Here are all the functions which use the rotfit.out files. '''

# Function to return rotfit parameter results from .out file header
def rotfitParams(rcdfile):
	with open(rcdfile, 'r') as f:
		galaxyName = f.readline().strip()
		line = f.readline().split()
		diskML = float(line[3])
		bulgeML = float(line[6])
		rc = float(line[8])
		vh = float(line[10])
		return galaxyName, diskML, bulgeML, rc, vh
	
# Function to calculate Vflat by taking the mean of the rotational velocities
# which are flat witin about 5%.
def Vflat(rcdfile):
	rcd = pd.read_table(rcdfile, skiprows=2, delim_whitespace=True)
	last = len(rcd.V_Rot) - 1
	v = [rcd.V_Rot[last], rcd.V_Rot[last-1]]
	Vavg = np.mean(v)
	for i in range(len(rcd.V_Rot)-3,-1,-1):
		if np.fabs(rcd.V_Rot[i] - Vavg)/Vavg <= 0.05:
			v.append(rcd.V_Rot[i])
			Vavg = np.mean(v)
		else:
			break
	return Vavg
			
# Function which calculates and returns the linear and scaled
# radial baryon mass fraction distribution.
def massFracDist(rcdfile):
	# extract galaxy name for sorting when plotting
	galaxyName = rotfitParams(rcdfile)[0]
	# read in the data, SKIP R=0
	rcd = pd.read_table(rcdfile, skiprows=[0,1,3], delim_whitespace=True)
	# calculate ratio of baryonic to total mass
	vstarsq = ((rcd.V_disk**2)+(rcd.V_bulge**2))
	vgassq = rcd.V_gas**2
	vobssq = rcd.V_Rot**2
	Mrat = (vgassq + vstarsq)/(vobssq)
	return galaxyName, rcd.Rad, Mrat
	
# Function which calculates the transition radius.
def rtrans(rcdfile):
	gal, r, Mrat = massFracDist(rcdfile)
	last = len(r) - 1
	if all(m > 0.5 for m in Mrat):
		return r[last]
		#return 0
	elif all(m < 0.5 for m in Mrat):
		return r[0]
		#return 0
	else:
		for i in range(len(r)):
		# interpolate to find rtrans
			if Mrat[i] >= 0.5 and Mrat[i+1] <= 0.5:
				dy = Mrat[i] - Mrat[i+1]
				dx = r[i] - r[i+1]
				slope = dy/dx
				int = Mrat[i] - slope*r[i]
				Rtrans = (0.5 - int)/slope
				return Rtrans
					
# Function to calculate disk fraction, baryon fraction and dynamical mass at 2.2h.
def twoPointTwoh(rcdfile, hkpc):
	# read in the data
	rcd = pd.read_table(rcdfile, skiprows=[0,1,3], delim_whitespace=True)
	# calculate ratio of baryonic to total mass
	vstarsq = ((rcd.V_disk**2)+(rcd.V_bulge**2))
	vgassq = rcd.V_gas**2
	vobssq = rcd.V_Rot**2
	Mrat = (vgassq + vstarsq)/(vobssq)
	for i in range(len(rcd.Rad)):
		if rcd.Rad[i] <= 2.2*hkpc and rcd.Rad[i+1] >= 2.2*hkpc:
			# calculate baryon fraction at 2.2h
			dr = rcd.Rad[i] - rcd.Rad[i+1]
			slopeMrat = (Mrat[i] - Mrat[i+1])/dr
			intMrat = Mrat[i] - slopeMrat*rcd.Rad[i]
			fracBaryh = slopeMrat*(2.2*hkpc) + intMrat
			# calculate star fraction at 2.2h
			slopevstar = (vstarsq[i] - vstarsq[i+1])/dr
			slopevobs = (vobssq[i] - vobssq[i+1])/dr
			intvstar = vstarsq[i] - slopevstar*rcd.Rad[i]
			intvobs = vobssq[i] - slopevobs*rcd.Rad[i]
			vstar_dynh = slopevstar*(2.2*hkpc) + intvstar
			vobs_dynh = slopevobs*(2.2*hkpc) + intvobs
			fracStarh = vstar_dynh/vobs_dynh
			# calculate dynamical mass at 2.2h
			Mdynh = (((2.2*hkpc)*1000.)*vobs_dynh)/(0.004302)
	return fracStarh, fracBaryh, Mdynh
	
# Function to calculate baryon fraction and dynamical mass at last measured radius
def MLast(rcdfile):
	# read in the data
	rcd = pd.read_table(rcdfile, skiprows=[0,1,3], delim_whitespace=True)
	# calculate ratio of baryonic to total mass
	vstarsq = ((rcd.V_disk**2)+(rcd.V_bulge**2))
	vgassq = rcd.V_gas**2
	vobssq = unumpy.uarray((rcd.V_Rot**2), (rcd.V_err**2))
	Mrat = (vgassq + vstarsq)/(vobssq)
	# last measured point
	last = len(rcd.Rad) - 1
	lastr = rcd.Rad[last]
	MratLast = Mrat[last]
	MdynLast = ((lastr*1000.)*vobssq[last])/(0.004302)
	return MratLast, MdynLast
	
# Calculates stellar fraction, baryon fraction and dynamical mass at R25.
def baryatR25(rcdfile, r25kpc):
	# read in the data
	rcd = pd.read_table(rcdfile, skiprows=[0,1,3], delim_whitespace=True)
	# calculate ratio of baryonic to total mass
	vstarsq = ((rcd.V_disk**2)+(rcd.V_bulge**2))
	vgassq = rcd.V_gas**2
	vobssq = rcd.V_Rot**2
	Mrat = (vgassq + vstarsq)/(vobssq)
	last = len(rcd.Rad) - 1
	if rcd.Rad[last] < r25kpc:
		fracStarl = vstarsq[last]/vobssq[last]
		#fracBaryl = MLast(rcdfile)[0]
		fracBaryl = Mrat[last]
		Mdynl = MLast(rcdfile)[1]
		return fracStarl, fracBaryl, Mdynl
	else:
		for i in range(len(rcd.Rad)):
			if rcd.Rad[i] <= r25kpc and rcd.Rad[i+1] >= r25kpc:
				dr = rcd.Rad[i] - rcd.Rad[i+1]
				# calculate baryon fraction at R25
				slopeMrat = (Mrat[i] - Mrat[i+1])/dr
				intMrat = Mrat[i] - slopeMrat*rcd.Rad[i]
				fracBary25 = slopeMrat*r25kpc + intMrat
				# calculate star fraction at R25
				slopevstar = (vstarsq[i] - vstarsq[i+1])/dr
				slopevobs = (vobssq[i] - vobssq[i+1])/dr
				intvstar = vstarsq[i] - slopevstar*rcd.Rad[i]
				intvobs = vobssq[i] - slopevobs*rcd.Rad[i]
				vstarr25 = slopevstar*r25kpc + intvstar
				vobsr25 = slopevobs*r25kpc + intvobs
				fracStar25 = vstarr25/vobsr25
				# calculate dynamical mass at R25
				Mdyn25 = ((r25kpc*1000.)*vobsr25)/(0.004302)
				return fracStar25, fracBary25, Mdyn25

# Function to calculate dynamical mass at rtrans.
def MdynRtrans(rcdfile):
	# read in the data
	rcd = pd.read_table(rcdfile, skiprows=[0,1,3], delim_whitespace=True)
	# calculate ratio of baryonic to total mass
	vstarsq = ((rcd.V_disk**2)+(rcd.V_bulge**2))
	vgassq = rcd.V_gas**2
	vobssq = unumpy.uarray((rcd.V_Rot**2), (rcd.V_err**2))
	Mrat = (vstarsq + vgassq)/vobssq
	if all(m > 0.5 for m in Mrat):
		return MLast(rcdfile)[1]
	elif all(m < 0.5 for m in Mrat):
		return Mrat[0]
	else:
		# set rtrans by calling rtrans function
		rt = rtrans(rcdfile)
		for i in range(len(rcd.Rad)):
			if rcd.Rad[i] <= rt and rcd.Rad[i+1] >= rt:
				slopevobs = (vobssq[i] - vobssq[i+1])/(rcd.Rad[i] - rcd.Rad[i+1])
				intvobs  = vobssq[i] - slopevobs*rcd.Rad[i]
				vobssqRt = slopevobs*rt + intvobs
				MdynRt = ((rt*1000.)*vobssqRt)/(0.004302)
				return MdynRt









	



