import os, sys
import os.path as path
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rcParams
plt.rc('font', family='serif')
rcParams['axes.labelsize'] = 38
rcParams['xtick.labelsize'] = 38
rcParams['ytick.labelsize'] = 38
rcParams['legend.fontsize'] = 32
rcParams['axes.titlesize'] = 42

absmag_sun = 3.24 # at 3.6 microns

galdf = pd.read_csv("for_pandas.csv", index_col=0)

# functional form for the model RC for a pseudo-isothermal spherical DM halo
def halo(r, *p):
        V_h, R_c = p
	return V_h*np.sqrt(1-((R_c/r)*(np.arctan(r/R_c))))

def total(X, *p):
        V_h, R_c, dML, bML = p
        r, vgas, vdisk, vbulge = X
        return np.sqrt((halo(r, V_h, R_c)**2.) + (vgas**2.) + \
                       (dML * (vdisk**2.)) + (bML * (vbulge**2.)))

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

def fithaloML(X, y, yerr, p_init=[150, 7, 0.5, 0.5]):
        r, vgas, vdisk, vbulge = X
        popt, pcov = curve_fit(total, (r, vgas, vdisk, vbulge), y, p_init, sigma=yerr, \
                               bounds=([0, 0, 0.3, 0.3], [500, 100, 0.8, 0.8]))
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

def fixedML(df, L):
        dml = float(raw_input('Enter disk M/L (enter 0 if no disk component): '))
        bml = float(raw_input('Enter bulge M/L (enter 0 if no bulge component): '))
        ML = (dml, bml)
        vdisk = df.V_DISK * np.sqrt((ML[0] * L[0]) / 10**9.)
        vbulge = df.V_BULGE * np.sqrt((ML[1] * L[1]) / 10**9.)
        vbary = np.sqrt((df.V_gas**2.) + (vdisk**2.) + (vbulge**2.))
        '''The rotational velocities (mass) attributed to the DM halo 
        are the residual velocities (mass) needed such that the sum in 
        quadrature of the baryon rotational velociites (mass) and the DM 
        halo velocities (mass) equal the observed rotational velocities (mass).'''
        vfit = np.sqrt((df.VROT**2.) - (vbary**2.)).dropna()[1:]
        rfit = df.RAD.reindex(vfit.index)
        vres_err = (df.VROT * df.V_ERR) / np.sqrt((df.VROT**2.) - (vbary**2.))
        vfit_err = vres_err.reindex(vfit.index)
        popt, perr, chi_sq, red_chi_sq = fithalo_lsq(rfit, vfit, vfit_err)
        return popt, perr, chi_sq, red_chi_sq, ML
        
def fitMLlsq(df, L):
        vfit = df.VROT[1:] # skip r=0
        rfit = df.RAD.reindex(vfit.index)
        vfit_err = df.V_ERR.reindex(vfit.index)
        vgas = df.V_gas.reindex(vfit.index)
        vdisk = (df.V_DISK * np.sqrt(L[0] / 10**9.)).reindex(vfit.index)
        vbulge = (df.V_BULGE * np.sqrt(L[1] / 10**9.)).reindex(vfit.index)
        popt, perr, chi_sq, red_chi_sq = fithaloML((rfit, vgas, vdisk, vbulge), vfit, vfit_err)
        if all(vdisk[1:] == 0):
                popt[2] = 0.
        if all(vbulge[1:] == 0):
                popt[3] = 0.
        ML = (popt[2], popt[3])
        return popt, perr, chi_sq, red_chi_sq, ML

def fitMLgrid(df, L):
        minchi = 99.999
        for dml in np.linspace(0.3, 0.8, 6): # 0.3 to 0.8 by 0.1 steps
                for bml in np.linspace(0.3, 0.8, 6):
                        vdisk = df.V_DISK * np.sqrt((dml * L[0]) / 10**9.)
                        vbulge = df.V_BULGE * np.sqrt((bml * L[1]) / 10**9.)
                        vbary = np.sqrt((df.V_gas**2.) + (vdisk**2.) + \
                                               (vbulge**2.))
                        if vbary.max() > df.VROT[:5].max():
                                break
                        else:
                                vfit = np.sqrt((df.VROT**2.) - \
                                               (vbary**2.)).dropna()[1:]
                                rfit = df.RAD.reindex(vfit.index)
                                vres_err = (df.VROT * df.V_ERR) / \
                                           np.sqrt((df.VROT**2.) - (vbary**2.))
                                vfit_err = vres_err.reindex(vfit.index)
                                popt, perr, chi_sq, red_chi_sq = fithalo_lsq(rfit, vfit, vfit_err)
                                if chi_sq < minchi:
                                        minchi = chi_sq
                                        minredchi = red_chi_sq
                                        if all(vdisk[1:] == 0):
                                                dml = 0.
                                        if all(vbulge[1:] == 0):
                                                bml = 0.
                                        ML = (dml, bml)
        return popt, perr, minchi, minredchi, ML

def main():
        fname = sys.argv[1]
        if os.path.exists(fname) == False:
                print 'ERROR: File %s does not exist.' % fname
                sys.exit(1)
        with open(fname, 'r') as f:
                galaxyName = f.readline().strip()
                line = f.readline().split()
                diskmag = float(line[0])
                bulgemag = float(line[1])
                HIflux = float(f.readline().strip())
                dMpc = float(f.readline().strip())
                VHIrad = float(f.readline().strip())
        rcdf = pd.read_table(fname, skiprows=6, delim_whitespace=True)

        rcdf['V_gas'] = rcdf.V_GAS * np.sqrt(1.4)
        Ldisk = 10**(0.4 * (absmag_sun - diskmag + 5 * np.log10((dMpc * 10**6.) / 10.)))
        Lbulge = 10**(0.4 * (absmag_sun - bulgemag + 5 * np.log10((dMpc * 10**6.) / 10.)))
        L = (Ldisk, Lbulge)

        print '\n =================================\n'
        print ' Welcome! Please select a halo fitting option for %s from the menu:' % galaxyName
        print '  0: Exit'
        print '  1: Fit halo with user specified fixed M/L'
        print '  2: Fit halo & M/L using non-linear least-squares method\n'
        #print '  3: Fit halo & M/L using grid search for M/L\n'
                
        method = int(raw_input())

        method_dict = {1: fixedML,
                       2: fitMLlsq,
                       3: fitMLgrid}

        if method in method_dict:
                popt, perr, chi_sq, red_chi_sq, ML = method_dict[method](rcdf, L)
        elif method == 0:
                sys.exit(0)
        else:
                while method not in method_dict:
                        print 'Not a valid entry. Try again.'
                        method = int(raw_input())
                popt, perr, chi_sq, red_chi_sq, ML = method_dict[method](rcdf, L)

        #popt, perr, chi_sq, red_chi_sq, ML = fixedML(rcdf, L, ML)
        #popt, perr, chi_sq, red_chi_sq, ML = fitMLgrid(rcdf, L)
        #popt, perr, chi_sq, red_chi_sq, ML = fitMLlsq(rcdf, L)              

        print '\n =================================\n'
        print ' Fit parameters for %s:' % galaxyName
        print ' Disk M/L = %.1f' % ML[0]
        print ' Bulge M/L = %.1f' % ML[1]
        print ' V_H = %0.f +/- %0.f km/s' % (popt[0], perr[0])
        print ' R_C = %.2f +/- %.2f kpc' % (popt[1], perr[1])
        print ' chi squared = %.3f, reduced chi squared = %.3f' % (chi_sq, red_chi_sq)

        # create new columns
        rcdf['V_disk'] = rcdf.V_DISK * np.sqrt((ML[0] * L[0]) / 10**9.)
        rcdf['V_bulge'] = rcdf.V_BULGE * np.sqrt((ML[1] * L[1]) / 10**9.)
        rcdf['V_bary'] = np.sqrt((rcdf.V_gas**2.) + (rcdf.V_disk**2.) + (rcdf.V_bulge**2.))
        rcdf['V_halo'] = halo(rcdf.RAD, popt[0], popt[1]) # model halo RC
        rcdf.set_value(0, 'V_halo', 0) # replace first row NaN with 0
        rcdf['V_tot'] = np.sqrt((rcdf.V_bary**2.) + (rcdf.V_halo**2.)) # total fit

        # add +/- 0.1 to M/L, if within limits 0.3 <= M/L <= 0.8
        if float('{0:.1f}'.format(ML[0])) < 0.8: # so that 0.79999099123 doesn't get past
                Vdisk_high = rcdf.V_DISK * np.sqrt(((ML[0] + 0.1) * L[0]) / 10**9.)
        else:
                Vdisk_high = rcdf['V_disk']
        if float('{0:.1f}'.format(ML[1])) < 0.8:
                Vbulge_high = rcdf.V_BULGE * np.sqrt(((ML[1] + 0.1) * L[1]) / 10**9.)
        else:
                Vbulge_high = rcdf['V_bulge']
        if float('{0:.1f}'.format(ML[0])) > 0.3:
                Vdisk_low = rcdf.V_DISK * np.sqrt(((ML[0] - 0.1) * L[0]) / 10**9.)
        else:
                Vdisk_low = rcdf['V_disk']
        if float('{0:.1f}'.format(ML[1])) > 0.3:
                Vbulge_low = rcdf.V_BULGE * np.sqrt(((ML[1] - 0.1) * L[1]) / 10**9.)
        else:
                Vbulge_low = rcdf['V_bulge']
        Vbary_high = np.sqrt((rcdf.V_gas**2.) + (Vdisk_high**2.) + (Vbulge_high**2.))
        Vbary_low = np.sqrt((rcdf.V_gas**2.) + (Vdisk_low**2.) + (Vbulge_low**2.))
        # run fithalo twice
        # upper limit M/L
        vfit_high = np.sqrt((rcdf.VROT**2.) - (Vbary_high**2.)).dropna()[1:]
        rfit_high = rcdf.RAD.reindex(vfit_high.index)
        vres_err_high = (rcdf.VROT * rcdf.V_ERR) / np.sqrt((rcdf.VROT**2.) - \
                                                           (Vbary_high**2.))
        vfit_err_high = vres_err_high.reindex(vfit_high.index)
        popt_high, perr_high, chi_sq_high, red_chi_sq_high = fithalo_lsq(rfit_high, vfit_high, vfit_err_high)
        Vhalo_high = halo(rcdf.RAD, popt_high[0], popt_high[1])
        Vhalo_high[0] = 0. # replace first row NaN with 0
        Vtot_high = np.sqrt((Vbary_high**2.) + (Vhalo_high**2.))
        # lower limit M/L
        vfit_low = np.sqrt((rcdf.VROT**2.) - (Vbary_low**2.)).dropna()[1:]
        rfit_low = rcdf.RAD.reindex(vfit_low.index)
        vres_err_low = (rcdf.VROT * rcdf.V_ERR) / np.sqrt((rcdf.VROT**2.) - \
                                                          (Vbary_low**2.))
        vfit_err_low = vres_err_low.reindex(vfit_low.index)
        popt_low, perr_low, chi_sq_low, red_chi_sq_low = fithalo_lsq(rfit_low, vfit_low, vfit_err_low)
        Vhalo_low = halo(rcdf.RAD, popt_low[0], popt_low[1])
        Vhalo_low[0] = 0. # replace first row NaN with 0
        Vtot_low = np.sqrt((Vbary_low**2.) + (Vhalo_low**2.))
        
        # get rid of old values and fix the column names for aesthetics
        rcdf = rcdf.drop(['V_GAS', 'V_DISK', 'V_BULGE'], axis=1)
        rcdf.rename(columns={'RAD': 'Rad', 'VROT': 'V_Rot', 'V_ERR': 'V_err'}, \
                    inplace=True)
        
        # radii for plotting
        Rbary = rcdf.Rad[rcdf['V_bary'].idxmax()]
        Rdyn = 2.2 * ((galdf['h_R'][galaxyName] / 206.265) * dMpc)
        R25 = ((galdf['D_25'][galaxyName] / 2.) / 206.265) * dMpc
                
        # Plotting

        fig, ax1 = plt.subplots(figsize=(25,15))
        ax = plt.gca()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which='major', width=3, length=20)
        ax.tick_params(which='minor', width=3, length=10)
        ax1.set_xlabel('Radius (arcsec)')
        ax1.set_ylabel('Velocity (km s$^{-1}$)')
        arcrad = (rcdf.Rad / (dMpc * 1000.)) * (180. / np.pi) * 3600.
        ax1.plot(arcrad, rcdf.V_Rot, 'ok', marker='None', ls='None')
        ax2 = ax1.twiny()
        ax = plt.gca()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which='major', width=3, length=20)
        ax.tick_params(which='minor', width=3, length=10)
        ax2.set_xlabel('Radius (kpc)')
        ax2.set_xlim(0, rcdf.Rad.max() * 1.15)
        ax2.set_ylim(0, rcdf.V_Rot.max() * 1.4)
        plt.figtext(0.1, 1, galaxyName, fontsize=48, va='top') # galaxy title
        plt.figtext(0.15, 0.85, 'D = %.1f Mpc' % dMpc, fontsize=38)
        plt.figtext(0.15, 0.8, 'fixed M/L', fontsize=38)
        plt.figtext(0.65, 0.85, 'R$_\mathrm{C}$ = %.1f $\pm$ %.1f kpc' % \
                    (popt[1], perr[1]), fontsize=38)
        plt.figtext(0.65, 0.8, 'V$_\mathrm{H}$ = %0.f $\pm$ %0.f km s$^{-1}$' % \
                    (popt[0], perr[0]), fontsize=38)
        last = len(rcdf.Rad) - 1
        xlab = rcdf.Rad.max() * 1.05
        # mark radii
        arrow_height = (rcdf.V_Rot.max() * 1.4) * 0.08
        plt.annotate('R$_\mathrm{bary}$', xy=(Rbary, 0.), xycoords='data', \
                     xytext=(Rbary, arrow_height), textcoords='data', \
                     arrowprops=dict(arrowstyle='simple', fc='k'), \
                     fontsize=38, ha='center')
        plt.annotate('2.2h$_\mathrm{R}$', xy=(Rdyn, 0.), xycoords='data', \
                     xytext=(Rdyn, arrow_height), textcoords='data', \
                     arrowprops=dict(arrowstyle='simple', fc='k'), \
                     fontsize=38, ha='center')
        if R25 < rcdf.Rad.max() * 1.15:
                plt.annotate('R$_{25}$', xy=(R25, 0.), xycoords='data', \
                             xytext=(R25, arrow_height), textcoords='data', \
                             arrowprops=dict(arrowstyle='simple', fc='k'), \
                             fontsize=38, ha='center')
        # observed RC
        first_HI = np.where(rcdf.Rad == VHIrad)[0]
        for i in range(0, first_HI):
                plt.plot(rcdf.Rad[i], rcdf.V_Rot[i], 'ok', ms=20, ls='None', \
                         fillstyle='none', mew=5)
        for j in range(first_HI, len(rcdf.Rad) - 1):
                plt.plot(rcdf.Rad[j], rcdf.V_Rot[j], 'ok', ms=20, ls='None')
        plt.errorbar(rcdf.Rad, rcdf.V_Rot, yerr=rcdf.V_err, color='k', lw=5, ls='None')
        # model gas RC
        plt.plot(rcdf.Rad, rcdf.V_gas, 'g-', ls='--', lw=5, dashes=(20, 20))
        plt.text(xlab, rcdf.V_gas[last], 'Gas', fontsize=38, va='center')
        # model stellar disk RC
        if all(rcdf.V_disk[1:] != 0):
                plt.figtext(0.4, 0.85, 'disc M/L = %.1f' % ML[0], fontsize=38)
                plt.plot(rcdf.Rad, rcdf.V_disk, 'm-', ls=':', lw=5, dashes=(5, 15))
                plt.text(xlab, rcdf.V_disk[last], 'Disc', fontsize=38, va='center')
                plt.fill_between(rcdf.Rad, Vdisk_high, Vdisk_low, color='m', alpha=0.3)
        # model stellar bulge RC
        if all(rcdf.V_bulge[1:] != 0):
                if all(rcdf.V_disk[1:] != 0):
                        plt.figtext(0.4, 0.8, 'bulge M/L = %.1f' % ML[1], fontsize=38)
                        plt.plot(rcdf.Rad, rcdf.V_bulge, 'c-', ls='-.', lw=5, \
                                 dashes=[20, 20, 5, 20])
                        plt.text(xlab, rcdf.V_bulge[last], 'Bulge', fontsize=38, \
                                 va='center')
                        plt.fill_between(rcdf.Rad, Vbulge_high, Vbulge_low, color='c', \
                                         alpha=0.3)
                else:
                        plt.figtext(0.4, 0.85, 'bulge M/L = %.1f' % ML[1], fontsize=38)
                        plt.plot(rcdf.Rad, rcdf.V_bulge, 'c-', ls='-.', lw=5, \
                                 dashes=[20, 20, 5, 20])
                        plt.text(xlab, rcdf.V_bulge[last], 'Bulge', fontsize=38, \
                                 va='center')
                        plt.fill_between(rcdf.Rad, Vbulge_high, Vbulge_low, color='c', \
                                         alpha=0.3)
        # model total baryon RC
        plt.plot(rcdf.Rad, rcdf.V_bary, 'b-', ls='--', lw=5, \
                 dashes=[20, 10, 20, 10, 5, 10])
        plt.text(xlab, rcdf.V_bary[last], 'Bary', fontsize=38, va='center')
        plt.fill_between(rcdf.Rad, Vbary_high, Vbary_low, color='b', alpha=0.3)
        # model DM halo RC
        xfine = np.linspace(rcdf.Rad.min(), rcdf.Rad.max())
        plt.plot(xfine, halo(xfine, popt[0], popt[1]), 'r-', lw=5)
        plt.text(xlab, rcdf.V_halo[last], 'Halo', fontsize=38, va='center')
        plt.fill_between(rcdf.Rad, Vhalo_high, Vhalo_low, color='r', alpha=0.3)
        # best fitting total
        plt.plot(rcdf.Rad, rcdf.V_tot, 'k-', ls='-', lw=5)
        plt.text(xlab, rcdf.V_tot[last], 'Total', fontsize=38, va='center')
        plt.fill_between(rcdf.Rad, Vtot_high, Vtot_low, color='k', alpha=0.3)

        #plt.plot(rfit, vfit, 'or')

        print '\n Would you like to proceed interactively?'
        print '  0: Exit'
        print '  1: Yes, show & interact with plots.'
        print '  2: No, write out data and save plot as pdf.'
                
        interact = int(raw_input())

        while interact not in [0, 1, 2]:
                print 'Not a valid entry. Try again.'
                interact = int(raw_input())

        if interact == 0:
                sys.exit(0)
        elif interact == 1:
                plt.show()
        elif interact == 2:
                galname = galaxyName.replace(" ", "")
                figfile = galname + '.fithalo.rcd.pdf'
                plt.savefig(figfile)
                with open(galname + '.fithalo.rcd.dat', 'w') as wf:
                        wf.write(galaxyName + '\n')
                        wf.write('Disk M/L = %.1f, Bulge M/L = %.1f, R_C = %.1f+/-%.2f kpc, V_H = %0.f+/-%0.f km/s\n' % (ML[0], ML[1], popt[1], perr[1], popt[0], perr[0]))
                        wf.write('chi squared = %.3f, reduced chi squared = %.3f\n' % (chi_sq, red_chi_sq))
                        rcdf.to_csv(wf, sep='\t', float_format='%.2f', index=False)
               
if __name__ == '__main__':
        main()
