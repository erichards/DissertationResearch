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

''' Creates scaling relation plots showing R25, scale length,
B-R and SFR vs. total baryon mass. '''

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

# Determine gas masses
MHI, MH2, Mgas = calcGasMass(D_LTF, obs.F_HI, obs.F_CO, obs.F_CO_err)
# Determine stellar masses
Lstar, Mstar = calcStellarMass(D_LTF, obs['m_3.6,tot'], 0.5)
# Calculate baryonic mass
Mbary = Mstar + Mgas

''' Begin plotting commands '''

fig = plt.figure(1)
# Top left: R25 vs. Mbary
ax1 = plt.subplot(221)
plt.xlabel('$M_\mathrm{bary}$ (M$_\mathrm{\odot}$)')
plt.ylabel('$R_{25}$ (kpc)')
# Linearize R25
R25kpc = (linearize(D_LTF, obs.D_25))/2.
R25kpc = unumpy.uarray(R25kpc, (R25kpc*0.05))
S0,Sb,Sc,Irr = loglog_morph_points_error(Mbary,R25kpc,obs['T-type'])
plt.legend([S0,Sb,Sc,Irr], ['S0','Sb','Sc','Irr'], loc='upper left', 
	numpoints=1, borderpad=0.2, handletextpad=0.2)

# Top right: B-R vs. Mbary
ax2 = plt.subplot(222)
plt.xlabel('$M_\mathrm{bary}$ (M$_\mathrm{\odot}$)')
plt.ylabel('($B-R$)$_0$')
# Determine extinction corrected magnitudes at R25
mB25_0 = extinctMag(obs.m_B, obs.m_B_err, obs.A_B)
mR25_0 = extinctMag(obs.m_R, obs.m_R_err, obs.A_R)
# Calculate extinction corrected colors at R25
BR25_0 = mB25_0 - mR25_0
logx_morph_points_error(Mbary,BR25_0,obs['T-type'])

# Bottom left: scale length vs. Mbary
ax3 = plt.subplot(223)
plt.xlabel('$M_\mathrm{bary}$ (M$_\mathrm{\odot}$)')
plt.ylabel('$h_\mathrm{R}$ (kpc)')
# Linearize h
hkpc = linearize(D_LTF, obs.h_R)
hkpc = unumpy.uarray(hkpc, (hkpc*0.05))
loglog_morph_points_error(Mbary,hkpc,obs['T-type'])
plt.ylim(0.0, 10.0)

# Bottom right: SFR vs. Mbary
ax4 = plt.subplot(224)
plt.xlabel('$M_\mathrm{bary}$ (M$_\mathrm{\odot}$)')
plt.ylabel('SFR (M$_\mathrm{\odot}$ yr$^{-1}$)')
# Determine star formation rates
sfr = calcSFR(D_LTF, obs['log(F_Ha)'], obs['log(F_Ha)_err'])
#ssfr = (sfr/Mstar)*(1E9)
loglog_morph_points_error(Mbary,sfr,obs['T-type'])

plt.tight_layout()
plt.savefig("ScalingRelationsMbary.pdf")
#plt.show()
plt.clf()