import sys
import os
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

#=============================================================================#

absmag_sun = 3.24 # at 3.6 microns

# functional form for the model RC for a pseudo-isothermal spherical DM halo
def halo(r, *p):
        V_H, R_C = p
	return V_H*np.sqrt(1-((R_C/r)*(np.arctan(r/R_C))))

def total(X, *p):
        V_H, R_C, dML, bML = p
        r, vgas, vdisk, vbulge = X
        return np.sqrt((halo(r, V_H, R_C)**2.) + (vgas**2.)
                       + (dML * (vdisk**2.)) + (bML * (vbulge**2.)))

''' curve_fit uses nonlinear least squares to fit a specified 
function to data. For bound problems it uses the trf 
(Trust Region Reflective) method for optimization. It defaults 
to Levenberg-Marquardt for unconstrained problems. Bounds are 
necessary for this problem to prevent NaNs in sqrt. '''
##### FOR FIXED M/L #####
def fithalo_lsq(x, y, yerr, p_init=[150, 7]):
        # doesn't work (OptimizeWarning) without limits: 0 < V_h,R_c < 500
        popt, pcov = curve_fit(halo, x, y, p_init, sigma=yerr, bounds=(0,500))
        perr = np.sqrt(np.diag(pcov))
        chi_sq = np.sum(((halo(x, *popt) - y) / yerr)**2.)
        red_chi_sq = chi_sq / (len(x) - len(popt))
        return popt, perr, chi_sq, red_chi_sq

##### FITS FOR M/L #####
def fithaloML(X, y, yerr, p_init=[150, 7, 0.5, 0.5]):
        r, vgas, vdisk, vbulge = X
        popt, pcov = curve_fit(total, (r, vgas, vdisk, vbulge),
                               y, p_init, sigma=yerr,
                               bounds=([0, 0, 0.3, 0.3], [500, 100, 0.8, 0.8]))
        perr = np.sqrt(np.diag(pcov))
        chi_sq = np.sum(((total(
                (r, vgas, vdisk, vbulge), *popt) - y) / yerr)**2.)
        red_chi_sq = chi_sq / (len(y) - len(popt))
        return popt, perr, chi_sq, red_chi_sq

##### FEEDS SPECIFIED M/L TO fithalo_lsq #####
def fixedML(r, V):
        vbary = np.sqrt((V[2]**2.) + (V[3]**2.) + (V[4]**2.))

        '''The rotational velocities (mass) attributed to the 
        DM halo are the residual velocities (mass) needed such 
        that the sum in quadrature of the baryon rotational 
        velociites (mass) and the DM halo velocities (mass) 
        equal the observed rotational velocities (mass).'''

        vfit = np.sqrt((V[0]**2.) - (vbary**2.)).dropna()[1:]
        rfit = r.reindex(vfit.index)
        vfit_err = ((V[0] * V[1]) / np.sqrt((V[0]**2.)
                                            - (vbary**2.))).reindex(vfit.index)
        #vfit_err = np.ones(len(vfit))
        popt, perr, chi_sq, red_chi_sq = fithalo_lsq(rfit, vfit, vfit_err)
        return popt, perr, chi_sq, red_chi_sq        
        
##### M/L AS FREE PARAMETER, FEEDS TO fithaloML #####
def fitMLlsq(r, V):
        vfit = V[0][1:] # skip r=0
        rfit = r.reindex(vfit.index)
        vfit_err = V[1].reindex(vfit.index)
        #vfit_err = np.ones(len(vfit))
        vgas = V[2].reindex(vfit.index)
        vdisk = V[3].reindex(vfit.index)
        vbulge = V[4].reindex(vfit.index)
        popt, perr, chi_sq, red_chi_sq = fithaloML(
                (rfit, vgas, vdisk, vbulge), vfit, vfit_err)
        if all(vdisk[1:] == 0):
                popt[2] = 0.
        if all(vbulge[1:] == 0):
                popt[3] = 0.
        ML = (popt[2], popt[3])
        return popt, perr, chi_sq, red_chi_sq, ML

def MLrange(r, V, dML, bML):
        # add +/- 0.1 to M/L, if within limits 0.3 <= M/L <= 0.8
        # format so that 0.79999099123 doesn't get past
        if float('{0:.1f}'.format(dML)) < 0.8:
                Vdisk_high = V[3] * np.sqrt((dML + 0.1) / dML)
        else:
                Vdisk_high = V[3]
        if float('{0:.1f}'.format(bML)) < 0.8:
                Vbulge_high = V[4] * np.sqrt((bML + 0.1) / bML)
        else:
                Vbulge_high = V[4]
        if float('{0:.1f}'.format(dML)) > 0.3:
                Vdisk_low = V[3] * np.sqrt((dML - 0.1) / dML)
        else:
                Vdisk_low = V[3]
        if float('{0:.1f}'.format(bML)) > 0.3:
                Vbulge_low = V[4] * np.sqrt((bML - 0.1) / bML)
        else:
                Vbulge_low = V[4]
        Vbary_high = np.sqrt((V[2]**2.) + (Vdisk_high**2.) + (Vbulge_high**2.))
        Vbary_low = np.sqrt((V[2]**2.) + (Vdisk_low**2.) + (Vbulge_low**2.))
        # run fithalo twice
        # upper limit M/L
        vfit_high = np.sqrt((V[0]**2.) - (Vbary_high**2.)).dropna()[1:]
        rfit_high = r.reindex(vfit_high.index)
        vfit_err_high = ((V[0] * V[1]) / np.sqrt(
                (V[0]**2.) - (Vbary_high**2.))).reindex(vfit_high.index)
        popt_high, perr_high, chi_sq_high, red_chi_sq_high = fithalo_lsq(
                rfit_high, vfit_high, vfit_err_high)
        Vhalo_high = halo(r, popt_high[0], popt_high[1])
        Vhalo_high[0] = 0. # replace first row NaN with 0
        Vtot_high = np.sqrt((Vbary_high**2.) + (Vhalo_high**2.))
        # lower limit M/L
        vfit_low = np.sqrt((V[0]**2.) - (Vbary_low**2.)).dropna()[1:]
        rfit_low = r.reindex(vfit_low.index)
        vfit_err_low = ((V[0] * V[1]) / np.sqrt(
                (V[0]**2.) - (Vbary_low**2.))).reindex(vfit_low.index)
        popt_low, perr_low, chi_sq_low, red_chi_sq_low = fithalo_lsq(
                rfit_low, vfit_low, vfit_err_low)
        Vhalo_low = halo(r, popt_low[0], popt_low[1])
        Vhalo_low[0] = 0. # replace first row NaN with 0
        Vtot_low = np.sqrt((Vbary_low**2.) + (Vhalo_low**2.))

        Vdisk_range = (Vdisk_high, Vdisk_low)
        Vbulge_range = (Vbulge_high, Vbulge_low)
        Vbary_range = (Vbary_high, Vbary_low)
        Vhalo_range = (Vhalo_high, Vhalo_low)
        Vtot_range = (Vtot_high, Vtot_low)

        return Vdisk_range, Vbulge_range, Vbary_range, Vhalo_range, Vtot_range

def find_nearest(radii, xvalue):
        return (np.abs(radii - xvalue)).argmin()

class InteractivePlot:
        def __init__(self, fig, txt, rc, df, ML):
                self.fig = fig
                self.dMLtxt = txt[0]
                self.bMLtxt = txt[1]
                self.VHtxt = txt[2]
                self.RCtxt = txt[3]
                self.disk = rc[0]
                self.bulge = rc[1]
                self.bary = rc[2]
                self.halo = rc[3]
                self.total = rc[4]
                self.r = df.RAD
                self.Vrot = df.VROT # V[0]
                self.Verr = df.V_ERR # V[1]
                self.Vgas = df.V_gas # V[2]
                self.Vdisk = df.V_disk # V[3]
                self.Vbulge = df.V_bulge # V[4]
                self.Vbary = df.V_bary # V[5]
                self.Vhalo = df.V_halo # V[6]
                self.dML = ML[0]
                self.bML = ML[1]
                self.toggle_on = False

        def connect(self):
                self.cidclick = self.fig.canvas.mpl_connect(
                        'button_press_event', self.on_click)
                self.cidkey = self.fig.canvas.mpl_connect(
                        'key_press_event', self.on_press)
                
        def disconnect(self):
                self.fig.canvas.mpl_disconnect(self.cidclick)
                self.fig.canvas.mpl_disconnect(self.cidkey)

        def change_bary(self):
                self.Vbary = np.sqrt((self.Vgas**2.)
                                     + (self.Vdisk**2.) + (self.Vbulge**2.))
                self.bary.set_data(self.r, self.Vbary)
                self.bary.figure.canvas.draw()

        def change_total(self):
                self.Vtotal = np.sqrt((self.Vbary**2.) + (self.Vhalo**2.))
                self.total.set_data(self.r, self.Vtotal)
                self.total.figure.canvas.draw()

        def change_disk(self):
                self.disk.set_data(self.r, self.Vdisk)
                self.disk.figure.canvas.draw()
                self.dMLtxt = plt.figtext(
                        0.4, 0.85, 'disc M/L = %.1f' % self.dML, fontsize=38)
                self.fig.canvas.draw()

        def change_bulge(self):
                self.bulge.set_data(self.r, self.Vbulge)
                self.bulge.figure.canvas.draw()
                self.bMLtxt = plt.figtext(
                        0.4, 0.8, 'bulge M/L = %.1f' % self.bML, fontsize=38)
                self.fig.canvas.draw()

        def change_halo(self):
                xfine = np.linspace(self.r.min(), self.r.max())
                self.halo.set_data(xfine, halo(xfine, self.V_H, self.R_C))
                self.halo.figure.canvas.draw()
                self.Vhalo = halo(self.r, self.V_H, self.R_C)
                self.RCtxt = plt.figtext(
                        0.65, 0.85, 'R$_\mathrm{C}$ = %.1f kpc'
                        % self.R_C, fontsize=38)
                self.VHtxt = plt.figtext(
                        0.65, 0.8, 'V$_\mathrm{H}$ = %0.f km s$^{-1}$'
                        % self.V_H, fontsize=38)
                self.fig.canvas.draw()

        def change_halo_fit(self):
                xfine = np.linspace(self.r.min(), self.r.max())
                self.halo.set_data(xfine, halo(xfine, self.V_H, self.R_C))
                self.halo.figure.canvas.draw()
                self.Vhalo = halo(self.r, self.V_H, self.R_C)
                self.RCtxt = plt.figtext(
                        0.65, 0.85, 'R$_\mathrm{C}$ = %.1f $\pm$ %.1f kpc'
                        % (self.R_C, self.R_C_err), fontsize=38)
                self.VHtxt = plt.figtext(
                        0.65, 0.8,
                        'V$_\mathrm{H}$ = %0.f $\pm$ %0.f km s$^{-1}$'
                        % (self.V_H, self.V_H_err), fontsize=38)
                self.fig.canvas.draw()

        def fit_halo(self):
                vels = (self.Vrot, self.Verr, self.Vgas,
                        self.Vdisk, self.Vbulge)
                popt, perr, self.chi_sq, self.red_chi_sq = fixedML(
                        self.r, vels)
                self.R_C = popt[1]
                self.R_C_err = perr[1]
                self.V_H = popt[0]
                self.V_H_err = perr[0]
                self.change_halo_fit()

        def remove_disk_text(self):
                self.dMLtxt.remove()

        def remove_bulge_text(self):
                self.bMLtxt.remove()

#        def remove_halo_text_no_err(self):
#                self.VHtxt.remove()
#                self.RCtxt.remove()

        def remove_halo_text(self):
                self.VHtxt.remove()
                self.RCtxt.remove()

        def on_click(self, event):
                if event.inaxes != self.disk.axes: return
                idx = find_nearest(self.r, event.xdata)
                if event.button == 1:
                        scale = event.ydata / self.Vdisk[idx]
                        self.Vdisk = scale * self.Vdisk
                        self.dML = (scale**2.) * self.dML
                        self.remove_disk_text()
                        self.change_disk()
                        self.change_bary()
                        self.change_total()
                elif event.button == 2:
                        scale = event.ydata / self.Vbulge[idx]
                        self.Vbulge = scale * self.Vbulge
                        self.bML = (scale**2.) * self.bML
                        self.remove_bulge_text()
                        self.change_bulge()
                        self.change_bary()
                        self.change_total()
                else:
                        self.R_C = event.xdata
                        self.V_H = event.ydata
                        self.remove_halo_text()
                        self.change_halo()
                        self.change_total()

        def on_press(self, event):
                vels = (self.Vrot, self.Verr, self.Vgas,
                        self.Vdisk, self.Vbulge)
                if event.key == 'e':
                        # get upper/lower limits from +/- 0.1 range in M/L
                        Vdisk_range, Vbulge_range, Vbary_range, Vhalo_range, Vtot_range = MLrange(self.r, vels, self.dML, self.bML)
                        if self.toggle_on == False:
                                self.toggle_on = True
                                self.disk_bands = plt.fill_between(
                                        self.r, Vdisk_range[0],
                                        Vdisk_range[1], color='m', alpha=0.3)
                                self.bulge_bands = plt.fill_between(
                                        self.r, Vbulge_range[0],
                                        Vbulge_range[1], color='c', alpha=0.3)
                                self.bary_bands = plt.fill_between(
                                        self.r, Vbary_range[0],
                                        Vbary_range[1], color='b', alpha=0.3)
                                self.halo_bands = plt.fill_between(
                                        self.r, Vhalo_range[0],
                                        Vhalo_range[1], color='r', alpha=0.3)
                                self.total_bands = plt.fill_between(
                                        self.r, Vtot_range[0],
                                        Vtot_range[1], color='k', alpha=0.3)
                                self.fig.canvas.draw()
                        else:
                                self.disk_bands.remove()
                                self.bulge_bands.remove()
                                self.bary_bands.remove()
                                self.halo_bands.remove()
                                self.total_bands.remove()
                                self.toggle_on = False
                elif event.key == 'h':
                        self.remove_halo_text()
                        self.fit_halo()
                        self.change_total()
                else: return

def draw_plot(meta, df, popt, perr, ML):
        gal, dMpc, VHIrad = meta
        galdf = pd.read_csv("for_pandas.csv", index_col=0)
        # radii for plotting
        Rbary = df.RAD[df['V_bary'].idxmax()]
        Rdyn = 2.2 * ((galdf['h_R'][gal] / 206.265) * dMpc)
        R25 = ((galdf['D_25'][gal] / 2.) / 206.265) * dMpc
                
        fig = plt.figure(figsize=(25,15), dpi=40)
        ax1 = fig.add_subplot(111)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.tick_params(which='major', width=3, length=20)
        ax1.tick_params(which='minor', width=3, length=10)
        ax1.set_xlabel('Radius (arcsec)')
        ax1.set_ylabel('Velocity (km s$^{-1}$)')
        arcrad = (df.RAD / (dMpc * 1000.)) * (206265.)
        ax1.set_xlim(0, arcrad.max() * 1.15)
        ax1.set_ylim(0, df.VROT.max() * 1.6)
        ax1.plot(arcrad, df.VROT, 'ok', marker='None', ls='None')
        ax2 = ax1.twiny()
        ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.tick_params(which='major', width=3, length=20)
        ax2.tick_params(which='minor', width=3, length=10)
        ax2.set_xlabel('Radius (kpc)')
        ax2.set_xlim(0, df.RAD.max() * 1.15)
        ax2.set_ylim(0, df.VROT.max() * 1.6)
        plt.figtext(0.1, 1, gal, fontsize=48, va='top') # galaxy title
        plt.figtext(0.15, 0.85, 'D = %.1f Mpc' % dMpc, fontsize=38)
        #plt.figtext(0.15, 0.8, 'max. disc', fontsize=38)
        RCtxt = plt.figtext(0.65, 0.85, 'R$_\mathrm{C}$ = %.1f $\pm$ %.1f kpc'
                            % (popt[1], perr[1]), fontsize=38)
        VHtxt = plt.figtext(
                0.65, 0.8, 'V$_\mathrm{H}$ = %0.f $\pm$ %0.f km s$^{-1}$'
                % (popt[0], perr[0]), fontsize=38)
        last = len(df.RAD) - 1
        xlab = df.RAD.max() * 1.05
        # mark radii
        arrow_height = (df.VROT.max() * 1.4) * 0.08
        plt.annotate('R$_\mathrm{bary}$', xy=(Rbary, 0.), xycoords='data',
                     xytext=(Rbary, arrow_height), textcoords='data', 
                     arrowprops=dict(arrowstyle='simple', fc='k'), 
                     fontsize=38, ha='center')
        plt.annotate('2.2h$_\mathrm{R}$', xy=(Rdyn, 0.), xycoords='data',
                     xytext=(Rdyn, arrow_height), textcoords='data',
                     arrowprops=dict(arrowstyle='simple', fc='k'),
                     fontsize=38, ha='center')
        if R25 < df.RAD.max() * 1.15:
                plt.annotate('R$_{25}$', xy=(R25, 0.), xycoords='data',
                             xytext=(R25, arrow_height), textcoords='data',
                             arrowprops=dict(arrowstyle='simple', fc='k'),
                             fontsize=38, ha='center')
        # observed RC
        first_HI = np.where(df.RAD == VHIrad)[0]
        for i in range(0, first_HI):
                plt.plot(df.RAD[i], df.VROT[i], 'ok', ms=20,
                         ls='None', fillstyle='none', mew=5)
        for j in range(first_HI, len(df.RAD)):
                plt.plot(df.RAD[j], df.VROT[j], 'ok', ms=20, ls='None')
        plt.errorbar(df.RAD, df.VROT, yerr=df.V_ERR,
                     color='k', lw=5, ls='None')
        # model gas RC
        plt.plot(df.RAD, df.V_gas, 'g-', ls='--', lw=5, dashes=(20, 20))
        plt.text(xlab, df.V_gas[last], 'Gas', fontsize=38, va='center')
        # model stellar disk RC
        if all(df.V_disk[1:] != 0):
                dMLtxt = plt.figtext(
                        0.4, 0.85, 'disc M/L = %.1f' % ML[0], fontsize=38)
                diskRC = plt.plot(
                        df.RAD, df.V_disk, 'm-', ls=':', lw=5, dashes=(5, 15))
                plt.text(xlab, df.V_disk[last], 'Disc',
                         fontsize=38, va='center')
        # model stellar bulge RC
        if all(df.V_bulge[1:] != 0):
                bulgeRC = plt.plot(df.RAD, df.V_bulge, 'c-',
                                   ls='-.', lw=5, dashes=[20, 20, 5, 20])
                plt.text(xlab, df.V_bulge[last], 'Bulge',
                         fontsize=38, va='center')
                if all(df.V_disk[1:] != 0):
                        bMLtxt = plt.figtext(0.4, 0.8, 'bulge M/L = %.1f'
                                             % ML[1], fontsize=38)
                else:
                        bMLtxt = plt.figtext(0.4, 0.85, 'bulge M/L = %.1f'
                                             % ML[1], fontsize=38)
        # model total baryon RC
        baryRC = plt.plot(df.RAD, df.V_bary, 'b-', ls='--', lw=5,
                          dashes=[20, 10, 20, 10, 5, 10])
        plt.text(xlab, df.V_bary[last], 'Bary', fontsize=38, va='center')
        # model DM halo RC
        xfine = np.linspace(df.RAD.min(), df.RAD.max())
        haloRC = plt.plot(xfine, halo(xfine, popt[0], popt[1]), 'r-', lw=5)
        plt.text(xlab, df.V_halo[last], 'Halo', fontsize=38, va='center')
        # best fitting total
        totRC = plt.plot(df.RAD, df.V_tot, 'k-', ls='-', lw=5)
        plt.text(xlab, df.V_tot[last], 'Total', fontsize=38, va='center')

        rcs = (diskRC[0], bulgeRC[0], baryRC[0], haloRC[0], totRC[0])
        txt = (dMLtxt, bMLtxt, VHtxt, RCtxt)

        return fig, txt, rcs

def main():
        # start by reading in the file
        if len(sys.argv) == 1:
                print 'Correct usage: "$ python fithalo.py path/filename.rot"'
                sys.exit(1)
        fname = sys.argv[1]
        if os.path.exists(fname) == False:
                print 'ERROR: File %s does not exist.' % fname
                sys.exit(1)
        with open(fname, 'r') as f:
                galaxyName = f.readline().strip()
                line = f.readline().split()
                diskmag = float(line[0])
                bulgemag = float(line[1])
                HIflux = float(f.readline().strip()) # old, no longer used
                dMpc = float(f.readline().strip())
                VHIrad = float(f.readline().strip()) # new addition to files

        # create data frame for rotation curves
        rcdf = pd.read_table(fname, skiprows=6, delim_whitespace=True)
        # store meta data to be fed into plotting function
        meta = (galaxyName, dMpc, VHIrad)

        # un-changing values
        rcdf['V_gas'] = rcdf.V_GAS * np.sqrt(1.4) # scale by 1.4 for He
        Ldisk = 10**(0.4 * (absmag_sun - diskmag
                            + 5 * np.log10((dMpc * 10**6.) / 10.)))
        Lbulge = 10**(0.4 * (absmag_sun - bulgemag
                             + 5 * np.log10((dMpc * 10**6.) / 10.)))
        # calculate disk & bulge rotation curves assuming M/L = 1
        rcdf['VdiskML1'] = rcdf.V_DISK * np.sqrt(Ldisk / 10**9.)
        rcdf['VbulgeML1'] = rcdf.V_BULGE * np.sqrt(Lbulge / 10**9.)

        # initial fit for halo & M/L's
        vels = (rcdf.VROT, rcdf.V_ERR, rcdf.V_gas,
                rcdf.VdiskML1, rcdf.VbulgeML1)
        popt, perr, chi_sq, red_chi_sq, ML = fitMLlsq(rcdf.RAD, vels)

        print '\n =================================\n'
        print ' Initial fit parameters for %s:\n' % galaxyName
        print ' Disk M/L =            %.1f' % ML[0]
        print ' Bulge M/L =           %.1f' % ML[1]
        print ' V_H =                 %0.f +/- %0.f km/s' % (popt[0], perr[0])
        print ' R_C =                 %.2f +/- %.2f kpc' % (popt[1], perr[1])
        print ' chi squared =         %.3f' % chi_sq
        print ' reduced chi squared = %.3f' % red_chi_sq
        
        # add columns based on fit parameters
        rcdf['V_disk'] = rcdf.VdiskML1 * np.sqrt(ML[0])
        rcdf['V_bulge'] = rcdf.VbulgeML1 * np.sqrt(ML[1])
        rcdf['V_bary'] = np.sqrt((rcdf.V_gas**2.)
                                 + (rcdf.V_disk**2.) + (rcdf.V_bulge**2.))
        rcdf['V_halo'] = halo(rcdf.RAD, popt[0], popt[1])
        rcdf.set_value(0, 'V_halo', 0) # replace first row NaN with 0
        rcdf['V_tot'] = np.sqrt((rcdf.V_bary**2.) + (rcdf.V_halo**2.))
        
        fig, txt, rcs = draw_plot(meta, rcdf, popt, perr, ML)
        
        ip = InteractivePlot(fig, txt, rcs, rcdf, ML)
        ip.connect()

        plt.show(block = False)

        print '\n =================================\n'
        print ' Welcome! Please select an option for %s from the menu:\n'% (
                galaxyName)
        print ' Plot options (use when plot window is active):'
        print '  mouse left-click:   adjust disk M/L'
        print '  mouse middle-click: adjust bulge M/L'
        print '  mouse right-click:  adjust halo fit parameters R_C & V_H'
        print '  e:                  toggle +/-0.1 M/L error bands on/off'
        print '  h:                  fit halo using current M/L\n'
        print ' Command line options'
        print ' --------------------'
        print ' Fit options:'
        print '  f:                  provide fixed M/L and re-fit halo'
        print '  r:                  re-fit halo & M/L using non-linear' 
        print '                         least-squares method'
        print ' Save options:'
        print '  p:                  save figure to file'
        print '  s:                  write rotation curves and fit' 
        print '                         parameters to text file'
        print '  q:                  exit without saving\n'
        print ' NOTE: You may have to click on the figure after pressing' 
        print '       "e" to toggle off the error bands. This is a known'
        print '        issue for some operating systems (Linux).\n'
                
        choice = raw_input()

        while choice not in ['f', 'r', 'p', 's', 'q']:
                print 'Not a valid entry. Please try again.'
                choice = raw_input()
        if choice == 'q':
                sys.exit(0)
        while choice != 'q':
                if choice == 'f':
                        ip.disconnect()
                        plt.close()
                        dml = float(raw_input('Enter disk M/L: '))
                        bml = float(raw_input('Enter bulge M/L: '))
                        ML = (dml, bml)
                        rcdf.V_disk = rcdf.VdiskML1 * np.sqrt(ML[0])
                        rcdf.V_bulge = rcdf.VbulgeML1 * np.sqrt(ML[1])
                        vels = (rcdf.VROT, rcdf.V_ERR, rcdf.V_gas,
                                    rcdf.V_disk, rcdf.V_bulge)
                        popt, perr, chi_sq, red_chi_sq = fixedML(
                                rcdf.RAD, vels)
                        rcdf['V_bary'] = np.sqrt(
                                (rcdf.V_gas**2.) + (rcdf.V_disk**2.)
                                + (rcdf.V_bulge**2.))
                        rcdf['V_halo'] = halo(rcdf.RAD, popt[0], popt[1])
                        rcdf.set_value(0, 'V_halo', 0) # replace 1st NaN with 0
                        rcdf['V_tot'] = np.sqrt((rcdf.V_bary**2.)
                                                + (rcdf.V_halo**2.))
                        fig, txt, rcs = draw_plot(meta, rcdf, popt, perr, ML)
                        ip = InteractivePlot(fig, txt, rcs, rcdf, ML)
                        ip.connect()
                        plt.show(block = False)
                        choice = raw_input()

                elif choice == 'r':
                        ip.disconnect()
                        plt.close()
                        vels = (rcdf.VROT, rcdf.V_ERR, rcdf.V_gas,
                                rcdf.VdiskML1, rcdf.VbulgeML1)
                        popt, perr, chi_sq, red_chi_sq, ML = fitMLlsq(
                                rcdf.RAD, vels)
                        rcdf['V_disk'] = rcdf.VdiskML1 * np.sqrt(ML[0])
                        rcdf['V_bulge'] = rcdf.VbulgeML1 * np.sqrt(ML[1])
                        rcdf['V_bary'] = np.sqrt(
                                (rcdf.V_gas**2.) + (rcdf.V_disk**2.)
                                + (rcdf.V_bulge**2.))
                        rcdf['V_halo'] = halo(rcdf.RAD, popt[0], popt[1])
                        rcdf.set_value(0, 'V_halo', 0) # replace 1st NaN with 0
                        rcdf['V_tot'] = np.sqrt((rcdf.V_bary**2.)
                                                + (rcdf.V_halo**2.))
                        fig, txt, rcs = draw_plot(meta, rcdf, popt, perr, ML)
                        ip = InteractivePlot(fig, txt, rcs, rcdf, ML)
                        ip.connect()
                        plt.show(block = False)
                        choice = raw_input()

                elif choice == 'p':
                        figfile = raw_input('Enter figure file name: ')
                        plt.savefig(figfile)
                elif choice == 's':
                        # get rid of old values & rename columns for aesthetics
                        rcdf = rcdf.drop(
                                ['V_GAS', 'V_DISK', 'V_BULGE'], axis=1)
                        rcdf.rename(columns = {'RAD': 'Rad',
                                               'VROT': 'V_Rot',
                                               'V_ERR': 'V_ERR'},
                                    inplace=True)
                        txtfile = raw_input('Enter text file name: ')
                        with open(txtfile, 'w') as wf:
                                wf.write(galaxyName + '\n')
                                wf.write('Disk M/L = %.1f, Bulge M/L = %.1f, R_C = %.1f+/-%.2f kpc, V_H = %0.f+/-%0.f km/s\n' % (ML[0], ML[1], popt[1], perr[1], popt[0], perr[0]))
                                wf.write('chi squared = %.3f, reduced chi squared = %.3f\n' % (chi_sq, red_chi_sq))
                                rcdf.to_csv(wf, sep='\t', float_format='%.2f', index=False)
                else: return
               
if __name__ == '__main__':
        main()
