import os, sys
import os.path as path
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rc('font', family='serif')
rcParams['axes.labelsize'] = 38
rcParams['xtick.labelsize'] = 38
rcParams['ytick.labelsize'] = 38
rcParams['legend.fontsize'] = 32
rcParams['axes.titlesize'] = 42

# functional form for the model RC for a pseudo-isothermal spherical DM halo
def halo(r, *p):
        V_h, R_c = p
	return V_h*np.sqrt(1-((R_c/r)*(np.arctan(r/R_c))))

def total(X, *p):
        V_h, R_c, dML, bML = p
        r, vgas, vdisk, vbulge = X
        return np.sqrt((halo(r, V_h, R_c)**2.) + (vgas**2.) + \
                       dML * (vdisk**2.) + bML * (vbulge**2.))

''' curve_fit uses nonlinear least squares to fit a specified function to data.
For bound problems it uses the trf (Trust Region Reflective) method for
optimization. It defaults to Levenberg-Marquardt for unconstrained problems.
Bounds are necessary for this problem to prevent NaNs in sqrt. '''
def fithalo_lsq(x, y, yerr, p_init=[150, 7]):
        # doesn't work (OptimizeWarning) without limits: 0 < V_h,R_c < 500
        popt, pcov = curve_fit(halo, x, y, p_init, sigma=yerr, bounds=(0,500))
        perr = np.sqrt(np.diag(pcov))
        chi_sq = np.sum(((halo(x, *popt) - y) / yerr)**2.)
        red_chi_sq = chi_sq / (len(x) - len(popt))
        return popt, perr, chi_sq, red_chi_sq

def fithaloML(X, y, yerr, p_init=[150, 7, 0.5, 1.0]):
        r, vgas, vdisk, vbulge = X
        popt, pcov = curve_fit(total, (r, vgas, vdisk, vbulge), y, p_init, sigma=yerr, \
                               bounds=([0, 0, 0.3, 0.3], [500, 100, 0.8, 1.0]))
        perr = np.sqrt(np.diag(pcov))
        chi_sq = np.sum(((total((r, vgas, vdisk, vbulge), *popt) - y) / yerr)**2.)
        red_chi_sq = chi_sq / (len(y) - len(popt))
        return popt, perr, chi_sq, red_chi_sq

'''Scipy's general nonlinear optimization routine minimizes any error
function (i.e. mean squared error or chi squared error) between the 
real data and specified model as a function of the model parameters 
(V_h & R_c in this case). I use Nelder-Mead method for unconstrained 
minimization, which uses the Simplex algorithm. I like it b/c it doesn't 
require or estimate the Jacobian (derivatives matrix).'''
# Minimize mean squared error
def fithalo_min_mse(x, y, yerr, p_init=[150, 7]):
        min_mse = lambda p: np.mean((halo(x, *p) - y)**2.)
        popt = minimize(min_mse, p_init, method='Nelder-Mead').x
        chi_sq = np.sum(((halo(x, *popt) - y) / yerr)**2.)
        red_chi_sq = chi_sq / (len(x) - len(popt))
        return popt, chi_sq, red_chi_sq

# Minimize chi squared (should produce same results as curve_fit)
# Or, alternatively, minimize Pearson's chi squared
def fithalo_min_chi(x, y, yerr, p_init=[150,7]):
        #min_chi = lambda p: np.sum(((halo(x, *p) - y) / yerr)**2.) # standard
        min_chi = lambda p: np.sum(((y - halo(x, *p))**2.) / halo(x, *p)) # Pearson's
        popt = minimize(min_chi, p_init, method='Nelder-Mead').x
        chi_sq = np.sum(((halo(x, *popt) - y) / yerr)**2.)
        red_chi_sq = chi_sq / (len(x) - len(popt))
        return popt, chi_sq, red_chi_sq

def main():
        # takes a directory of files; path given via command line
        fpath = sys.argv[1]
        #fpath = "test"
        if os.path.exists(fpath) == False:
                print 'ERROR: Directory %s does not exist.' % fpath
                sys.exit(1)
        flist = os.listdir(fpath)
        print flist
        for fname in flist:
                with open(fpath+'/'+fname, 'r') as f:
                        galaxyName = f.readline().strip()
                rcd = pd.read_table(fpath+'/'+fname, skiprows=2, delim_whitespace=True)
                rcd.V_disk = rcd.V_disk / np.sqrt(0.5)
                rcd.V_bulge = rcd.V_bulge / np.sqrt(0.4)
                # total model baryon RC
                #rcd['V_bary'] = np.sqrt((rcd.V_gas**2.) + (rcd.V_disk**2.) + \
                #                (rcd.V_bulge**2.))
                '''The rotational velocities (mass) attributed to the DM halo 
                are the residual velocities (mass) needed such that the sum in 
                quadrature of the baryon rotational velociites (mass) and the DM 
                halo velocities (mass) equal the observed rotational velocities (mass).'''
                #V_fit = np.sqrt((rcd.V_Rot**2.) - (rcd.V_bary**2.)).dropna()[1:]
                v_fit = rcd.V_Rot[1:] # skip r=0
                r_fit = rcd.Rad.reindex(v_fit.index)
                #vres_err = (rcd.V_Rot * rcd.V_err) / np.sqrt((rcd.V_Rot**2.) - \
                #                                             (rcd.V_bary**2.))
                v_fit_err = rcd.V_err.reindex(v_fit.index)
                vgas = rcd.V_gas.reindex(v_fit.index)
                vdisk = rcd.V_disk.reindex(v_fit.index)
                vbulge = rcd.V_bulge.reindex(v_fit.index)

                #p_init = [250, 5] # initial guesses for model paramters V_h & R_c
                #popt, perr, chi_sq, red_chi_sq = fithalo_lsq(r_fit, v_fit, v_fit_err)
                #popt, chi_sq, red_chi_sq = fithalo_min_mse(r_fit, v_fit, v_fit_err)
                #popt, chi_sq, red_chi_sq = fithalo_min_chi(r_fit, v_fit, v_fit_err)
                popt, perr, chi_sq, red_chi_sq = fithaloML((r_fit, vgas, vdisk, vbulge), v_fit, v_fit_err)
                print popt, perr, chi_sq, red_chi_sq

                rcd['V_halo'] = halo(rcd.Rad, popt[0], popt[1]) # model halo RC
                rcd.set_value(0, 'V_halo', 0) # replace first row NaN with 0
                #rcd['V_tot'] = total(rcd, popt[0], popt[1], 1.0, 1.0)
                rcd['V_tot'] = total((rcd.Rad, rcd.V_gas, rcd.V_disk, rcd.V_bulge), *popt)
                rcd.set_value(0, 'V_tot', 0) # replace first row NaN with 0
                #rcd['V_tot'] = np.sqrt((rcd.V_bary**2.) + (rcd.V_halo**2.)) # total fit
                rcd['V_disk'] = rcd.V_disk * popt[2]
                rcd['V_bulge'] = rcd.V_bulge * popt[3]
                rcd['V_bary'] = np.sqrt((rcd.V_gas**2.) + (rcd.V_disk**2.) + \
                                        (rcd.V_bulge**2.))
                 
                # Plotting
                 
                fig, ax1 = plt.subplots(figsize=(25,15))
                ax = plt.gca()
                ax.tick_params(width=3, length=20)
                ax1.set_xlabel('Radius (arcsec)')
                ax1.set_ylabel('Velocity (km s$^{-1}$)')
                dist = 10. # set dummy distance of 10 Mpc for now
                arcrad = (rcd.Rad / dist * 1000.) * (180. / np.pi) * 3600.
                ax2 = ax1.twiny()
                ax = plt.gca()
                ax.tick_params(width=3, length=20)
                ax2.set_xlabel('Radius (kpc)')
                ax2.set_xlim(0, rcd.Rad.max() * 1.15)
                ax2.set_ylim(0, rcd.V_Rot.max() * 1.35)
                plt.figtext(0.1, 1, galaxyName, fontsize=48, va='top') # galaxy title
                plt.figtext(0.15, 0.85, 'D = %2.1f Mpc' % dist, fontsize=38)
                plt.figtext(0.15, 0.8, 'fixed M/L', fontsize=38)
                plt.figtext(0.65, 0.85, 'R$_\mathrm{C}$ = %3.1f $\pm$ %3.1f kpc' % \
                            (popt[1], perr[1]), fontsize=38)
                plt.figtext(0.65, 0.8, 'V$_\mathrm{H}$ = %3.f $\pm$ %3.f km s$^{-1}$' % \
                            (popt[0], perr[0]), fontsize=38)
                last = len(rcd.Rad) - 1
                xlab = rcd.Rad.max() * 1.05
                # observed RC          
                plt.plot(rcd.Rad, rcd.V_Rot, 'ok', ms=20, ls='None')
                plt.errorbar(rcd.Rad, rcd.V_Rot, yerr=rcd.V_err, color='k', \
                             lw=5, ls='None')
                # model gas RC
                plt.plot(rcd.Rad, rcd.V_gas, 'g-', ls='--', lw=5, dashes=(20, 20))
                plt.text(xlab, rcd.V_gas[last], 'Gas', fontsize=38, va='center')
                # model stellar disk RC
                if all(rcd.V_disk[1:] != 0):
                        dML = popt[2]
                        plt.figtext(0.4, 0.85, 'disc M/L = %1.1f' % dML, fontsize=38)
                        plt.plot(rcd.Rad, rcd.V_disk, 'm-', ls=':', lw=5, \
                                 dashes=(5, 15))
                        plt.text(xlab, rcd.V_disk[last], 'Disc', fontsize=38, \
                                 va='center')
                # model stellar bulge RC
                if all(rcd.V_bulge[1:] != 0):
                        bML = popt[3]
                        if all(rcd.V_disk[1:] != 0):
                                plt.figtext(0.4, 0.8, 'bulge M/L = %1.1f' % bML, \
                                            fontsize=38)
                        else:
                                plt.figtext(0.4, 0.85, 'bulge M/L = %1.1f' % bML, \
                                            fontsize=38)
                        plt.plot(rcd.Rad, rcd.V_bulge, 'c-', ls='-.', lw=5, \
                                 dashes=[20, 20, 5, 20])
                        plt.text(xlab, rcd.V_bulge[last], 'Bulge', fontsize=38, \
                                 va='center')
                # model totaly baryon RC
                plt.plot(rcd.Rad, rcd.V_bary, 'b-', ls='--', lw=5, \
                         dashes=[20, 10, 20, 10, 5, 10])
                plt.text(xlab, rcd.V_bary[last], 'Bary', fontsize=38, va='center')
                # model DM halo RC
                xfine = np.linspace(rcd.Rad.min(), rcd.Rad.max())
                plt.plot(xfine, halo(xfine, popt[0], popt[1]), 'r-', lw=5)
                plt.text(xlab, rcd.V_halo[last], 'Halo', fontsize=38, va='center')
                # best fitting total
                plt.plot(rcd.Rad, rcd.V_tot, 'k-', ls='-', lw=5)
                plt.text(xlab, rcd.V_tot[last], 'Total', fontsize=38, va='center')

                #plt.plot(r_fit, V_res, 'or')

                plt.show()
                #plt.savefig("n5005rcdtest.png")

if __name__ == '__main__':
        main()
