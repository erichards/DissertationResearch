import pandas as pd
import numpy as np
from uncertainties import unumpy as un
import matplotlib as mpl
import matplotlib.pyplot as plt

''' Convenient functions to create scatter plots which
create different symbol point types based on a third parameter
(usually morphological T-type). '''

def logx_morph_points(x,y,morph):
    for i in range(len(morph)):
        if morph[i] == -1 or morph[i] == -2 or morph[i] == 0:
            S0, = plt.semilogx(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'ko',mfc='None')
        elif morph[i] == 1 or morph[i] == 2:
            Sa, = plt.semilogx(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'kD')
        elif morph[i] == 3 or morph[i] == 4:
            Sb, = plt.semilogx(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'ks')
        elif morph[i] == 5 or morph[i] == 6 or morph[i] == 7:
            Sc, = plt.semilogx(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'ko')
        else:
            Irr, = plt.semilogx(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'k*',markersize=10)
	return S0, Sb, Sc, Irr
	
def logx_morph_points_error(x,y,morph):
	for i in range(len(morph)):
		if morph[i] == -1 or morph[i] == -2 or morph[i] == 0:
			S0, = plt.semilogx(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'ko',mfc='None')
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='ko',mfc='None')
		elif morph[i] == 1 or morph[i] == 2:
			Sa, = plt.semilogx(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'kD')
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='kD')
		elif morph[i] == 3 or morph[i] == 4:
			Sb, = plt.semilogx(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'ks')
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='ks')
		elif morph[i] == 5 or morph[i] == 6 or morph[i] == 7:
			Sc, = plt.semilogx(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'ko')
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='ko')
		else:
			Irr, = plt.semilogx(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'k*',mfc='None',markersize=10)
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='k*',mfc='None',markersize=10)
	return S0, Sb, Sc, Irr
	
def loglog_morph_points_error(x,y,morph):
	for i in range(len(morph)):
		if morph[i] == -1 or morph[i] == -2 or morph[i] == 0:
			S0, = plt.loglog(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'ko',mfc='None')
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='ko',mfc='None')
		elif morph[i] == 1 or morph[i] == 2:
			Sa, = plt.loglog(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'kD')
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='kD')
		elif morph[i] == 3 or morph[i] == 4:
			Sb, = plt.loglog(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'ks')
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='ks')
		elif morph[i] == 5 or morph[i] == 6 or morph[i] == 7:
			Sc, = plt.loglog(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'ko')
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='ko')
		else:
			Irr, = plt.loglog(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'k*',mfc='None',markersize=10)
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='k*',mfc='None',markersize=10)
	return S0, Sb, Sc, Irr

def morph_points_error(x,y,morph):
	for i in range(len(morph)):
		if morph[i] == -1 or morph[i] == -2 or morph[i] == 0:
			S0, = plt.plot(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'ko',mfc='None')
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='ko',mfc='None')
		elif morph[i] == 1 or morph[i] == 2:
			Sa, = plt.plot(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'kD')
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='kD')
		elif morph[i] == 3 or morph[i] == 4:
			Sb, = plt.plot(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'ks')
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='ks')
		elif morph[i] == 5 or morph[i] == 6 or morph[i] == 7:
			Sc, = plt.plot(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'ko')
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='ko')
		else:
			Irr, = plt.plot(un.nominal_values(x[i]),
				un.nominal_values(y[i]),'k*',mfc='None',markersize=10)
			plt.errorbar(un.nominal_values(x[i]),
				un.nominal_values(y[i]),xerr=un.std_devs(x[i]),
				yerr=un.std_devs(y[i]),fmt='k*',mfc='None',markersize=10)
	return S0, Sb, Sc, Irr