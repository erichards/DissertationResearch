import pandas as pd
import numpy as np
from uncertainties import unumpy
from uncertainties import ufloat
import matplotlib as mpl
import matplotlib.pyplot as plt
from plot_morph_points import *
from rcd_results import *
from calculateParams import *
# stuff for pretty plots
from matplotlib import rcParams
plt.rc('font', family='serif')
plt.rc('font', serif='Computer Modern Roman')
plt.rc('text',usetex='true')
# This just changes the font size of different things
rcParams['axes.labelsize'] = 15
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['legend.fontsize'] = 15
rcParams['lines.markersize'] = 8

''' Creates plot showing baryon fraction at R25 vs. stellar properties
B-R color, B-R gradient, EW and EW gradient. '''

# Read in obs data -- skipping some rows; using only
# paper II galaxies for now...
obs = pd.read_csv("for_pandas.csv",skiprows=[2,5,7,13,15,17,18])

# Read in rotfit.out files
with open("fixMLFiles.txt", 'r') as f:
	fixMLFiles = f.read().splitlines()
	
# Calculate the Luminous Tully-Fisher distance
logLNIR_VTF = -0.28 + 4.93*(np.log10(obs.Vflat))
MNIR_VTF = 3.24 - 2.5*(logLNIR_VTF)
D_LTF = (10**(((obs['m_3.6,tot'] - MNIR_VTF)/5.) + 1.))/(1E6)

# Linearize R25
r25 = unumpy.uarray((obs.D_25/2.), ((obs.D_25/2.)*0.05))
r25kpcDind = linearize(obs.DMpc, r25)
r25kpcDltf = linearize(D_LTF, r25)

# Create list of stellar and baryon fraction measurements
# using baryatR25 function
fracStar25Dltf = []
fracBary25Dltf = []
for file in fixMLFiles:
	gal = rotfitParams(file)[0]
	for i in range(len(obs.Galaxy)):
		if gal == obs.Galaxy[i]:
			r25kpcGal = r25kpcDltf[i]
	fracStar25Dltf.append(baryatR25(file, r25kpcGal)[0])
	fracBary25Dltf.append(baryatR25(file, r25kpcGal)[1])

''' Begin plotting commands '''

plt.figure(1)
# Top left: baryon fraction at R25 vs. B-R color (also at R25)
ax1 = plt.subplot(221)
plt.xlabel('($B-R$)$_0$')
plt.ylabel('$V^{2}_\mathrm{bary}$/$V^{2}_\mathrm{tot}$ $@$ $R_{25}$')
plt.axhline(y=0.5, color='black', linewidth=2, linestyle='dashed')
# B-R color
# Determine extinction corrected magnitudes at R25
mB25_0 = extinctMag(obs.m_B, obs.m_B_err, obs.A_B)
mR25_0 = extinctMag(obs.m_R, obs.m_R_err, obs.A_R)
# Calculate extinction corrected colors at R25
BR25_0 = mB25_0 - mR25_0
S0,Sb,Sc,Irr = morph_points_error(BR25_0, fracBary25Dltf, obs['T-type'])
plt.legend([S0,Sb,Sc,Irr], ['S0','Sb','Sc','Irr'], loc='upper left', 
	numpoints=1, borderpad=0.2, handletextpad=0.2)
plt.xlim(0.6, 1.5)
plt.ylim(0.1, 1.1)
	
# Top right: baryon fraction at R25 vs. B-R color gradient
ax2 = plt.subplot(222)
plt.xlabel('($B-R$) gradient (arcmin$^{-1}$)')
#plt.ylabel('$V^{2}_\mathrm{bary}$/$V^{2}_\mathrm{tot}$ $@$ $R_{25}$')
plt.axhline(y=0.5, color='black', linewidth=2, linestyle='dashed')
#plt.axvline(x=0.0, color='black', linewidth=2, linestyle='dashed')
ax2.set_yticklabels([])
BRgrad = unumpy.uarray(obs['(B-R)_grad'], (unumpy.fabs(obs['(B-R)_grad']*0.05)))
morph_points_error(BRgrad, fracBary25Dltf, obs['T-type'])
#plt.xlim(-0.15, 0.05)
plt.ylim(0.1, 1.1)

# Bottom/middle left: baryon fraction at R25 vs. EW
ax3 = plt.subplot(223)
plt.xlabel('$\log_{10}$(EW) ($\AA$)')
plt.ylabel('$V^{2}_\mathrm{bary}$/$V^{2}_\mathrm{tot}$ $@$ $R_{25}$')
plt.axhline(y=0.5, color='black', linewidth=2, linestyle='dashed')
#ax3.set_yticklabels([])
EW = unumpy.uarray(obs.EW, obs.EW_err)
logEW = unumpy.log10(EW)
morph_points_error(logEW, fracBary25Dltf, obs['T-type'])
plt.xlim(0.2, 1.7)
plt.ylim(0.1, 1.1)

# Bottom/middle right: baryon fraction vs. EW gradient
ax4 = plt.subplot(224)
plt.xlabel('$\log_{10}$(EW) gradient (arcmin$^{-1}$)')
#plt.ylabel('$V^{2}_\mathrm{bary}$/$V^{2}_\mathrm{tot}$ $@$ $R_{25}$')
plt.axhline(y=0.5, color='black', linewidth=2, linestyle='dashed')
#plt.axvline(x=0.0, color='black', linewidth=2, linestyle='dashed')
ax4.set_yticklabels([])
logEWgrad = unumpy.uarray(obs['log(EW)_grad'], unumpy.fabs((obs['log(EW)_grad']*0.05)))
morph_points_error(logEWgrad, fracBary25Dltf, obs['T-type'])
plt.xlim(-1.6, 0.9)
plt.ylim(0.1, 1.1)

''' For later...
# Bottom left: baryon fraction vs. SFR
ax5 = plt.subplot(325)
plt.xlabel('SFR (M$_\mathrm{\odot}$ yr$^{-1}$)')
plt.ylabel('$V^{2}_\mathrm{bary}$/$V^{2}_\mathrm{tot}$ $@$ $R_{25}$')
plt.axhline(y=0.5, color='black', linewidth=2, linestyle='dashed')
sfr = calcSFR(D_LTF, obs['log(F_Ha)'], obs['log(F_Ha)_err'])
morph_points_error(sfr, fracBary25Dltf, obs['T-type'])
plt.xlim(-0.1, 3.9)
plt.ylim(0.1, 1.1)

# Bottom right: baryon fraction vs. C28
ax6 = plt.subplot(326)
plt.xlabel('$C_{28}$')
plt.axhline(y=0.5, color='black', linewidth=2, linestyle='dashed')
ax6.set_yticklabels([])
c28 = unumpy.uarray(obs.C_28, (obs.C_28)*0.05)
morph_points_error(c28, fracBary25Dltf, obs['T-type'])
plt.xlim(2.1, 6.3)
plt.ylim(0.1, 1.1)
'''

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.savefig("baryFracStarProps.pdf")
#plt.show()
plt.clf()
