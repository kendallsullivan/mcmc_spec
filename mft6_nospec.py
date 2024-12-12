#Kendall Sullivan

import numpy as np
import synphot
import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt
from astropy.io import fits
import os 
from glob import glob
from astropy import units as u
import time, sys, getopt
from PyAstronomy import pyasl
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import emcee
import corner
import extinction
import multiprocessing as mp
import matplotlib.cm as cm
import pyphot; lib = pyphot.get_library()
from dustmaps.bayestar import BayestarQuery
bayestar = BayestarQuery(version='bayestar2019')
from astropy.coordinates import SkyCoord

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def bccorr(wl, bcvel, radvel):
	"""Calculates a barycentric velocity correction given a barycentric and/or a radial velocity (set any unused value to zero)

	Args: 
		wl (array): wavelength vector.
		bcvel (float): a barycentric or heliocentric velocity in km/s
		radvel (float): a systemic radial velocity in km/s

	Returns: 
		a wavelength vector corrected for barycentric and radial velocities.

	"""
	return np.array(wl) * (1. + (bcvel - radvel)/3e5)

def extinct(wl, spec, av, rv = 3.1, unit = 'aa'):
	"""Uses the package "extinction" to calculate an extinction curve for the given A_v and R_v, 
	then converts the extinction curve to a transmission curve
	and uses that to correct the spectrum appropriately.
	Accepted units are angstroms ('aa', default) or microns^-1 ('invum').

	Args:
		wl (array): wavelength array
		spec (array): flux array
		av (float): extinction in magnitudes
		rv (float): Preferred R_V, defaults to 3.1
		unit (string): Unit to use. Accepts angstroms "aa" or inverse microns "invum". Defaults to angstroms.

	Returns:
		spec (list): a corrected spectrum vwith no wavelength vector. 
	"""
	ext_mag = extinction.ccm89(wl, av, rv)
	spec = extinction.apply(ext_mag, spec)
	return np.array(spec)

def get_radius(teff, matrix):
	#assume I'm using MIST 5 gyr model

	#get the age column
	aage = matrix[:, 1]

	#then get the correct temperatures and luminosities for a 1 gyr model (need this age bc the hot stars die pretty soon after that)
	teff5, lum5 = matrix[:,4][np.where(np.array(aage) == 9.0000000000000000)], matrix[:,6][np.where(np.array(aage) == 9.0000000000000000)]

	#interpolate to get the right luminosity for the temperature
	intep = interp1d(teff5[:220], lum5[:220]); lum = intep(teff)

	#define some constants because astropy.constants is probably too slow for this
	sigma_sb = 5.670374e-5 #erg/s/cm^2/K^4
	lsun = 3.839e33 #erg/s 
	rsun = 6.957e10 #cm
	#calculate the radius using the Stefan-Boltzmann law
	rad = np.sqrt(lum*lsun/(4 * np.pi * sigma_sb * teff**4))/rsun #solar radii
	#and return it
	return rad

def get_logg(teff, matrix):
	#get the correct log(g) from the models by interpolating in terms of temperature
	#get the age vector
	aage = matrix[:, 1]
	#now get the correct temperature and log(g) vectors for the desired age
	teff5 = matrix[:,4][np.where(np.array(aage) == 9.0000000000000000)]
	logg5 = matrix[:,5][np.where(np.array(aage) == 9.0000000000000000)]
	#interpolate to get the log(g) in terms of teff
	intep2 = interp1d(teff5[:220], logg5[:220]); log = intep2(teff)

	#and return it
	return log

def find_nearest(array, value):
	"""finds index in array such that the array component at the returned index is closest to the desired value.
	
	Args: 
		array (list): Array to search.
		value (float or int): Value to find closest value to.

	Returns: 
		idx (int): index at which array is closest to value

	"""
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

def chisq(model, data, var):
	#calculate chi square value of an model vs. data with a given variance
	#make sure that the two arrays are comparable
	#explicitly cast everything 
	try:
		return ((np.array(model) - np.array(data))**2)/np.array(var)**2
	except:
		print('Data and model must be the same length')

def broaden(even_wl, modelspec_interp, res, vsini = 0, limb = 0, plot = False):
	"""Adds resolution, vsin(i) broadening, taking into account limb darkening.
	"""
	#do the instrumental broadening and truncate the ends because they get messy
	broad = pyasl.instrBroadGaussFast(even_wl, modelspec_interp, res, maxsig=5)
	broad[0:5] = broad[5] 
	broad[len(broad)-10:len(broad)] = broad[len(broad) - 11]

	#if I want to impose stellar parameters of v sin(i) and limb darkening, do that here
	if vsini != 0 and limb != 0:
		rot = pyasl.rotBroad(even_wl, broad, limb, vsini)#
	#otherwise just move on
	else:
		rot = broad

	#Make a plotting option just in case I want to double check that this is doing what it's supposed to
	if plot == True:

		plt.figure()
		plt.plot(even_wl, sp, label = 'model')
		plt.plot(even_wl, broad, label = 'broadened')
		plt.plot(even_wl, rot, label = 'rotation')
		plt.legend(loc = 'best')
		plt.xlabel('wavelength (angstroms)')
		plt.ylabel('normalized flux')
		plt.savefig('rotation.pdf')

	#return the wavelength array and the broadened flux array
	return np.array(even_wl), np.array(rot)

def redres(wl, spec, factor):
	"""Imposes instrumental resolution limits on a spectrum and wavelength array
	Assumes evenly spaced wl array

	"""
	#decide the step size by using the median original wl spacing and then increase it by the appropriate factor
	new_stepsize = np.median([wl[n] - wl[n-1] for n in range(1, len(wl))]) * factor

	#make a new wl array using the new step size
	wlnew = np.arange(min(wl), max(wl), new_stepsize)

	#interpolate the spectrum so it's on the new wavelength scale 
	i = interp1d(wl, spec)
	specnew = i(wlnew)

	#return the reduced spectrum and wl array
	return np.array(wlnew), np.array(specnew)

def make_reg(wl, flux, waverange):
	"""given some wavelength range as an array, output flux and wavelength vectors within that range.

	Args:
		wl (list): wavelength array
		flux (list): flux array
		waverange (list): wavelength range array

	Returns: 
		wavelength and flux vectors within the given range

	"""
	#interpolate the input spectrum
	wl_interp = interp1d(wl, flux)
	#make a new wavelength array that's evenly spaced with the minimum spacing
	wlslice = np.arange(min(waverange), max(waverange), wl[1]-wl[0])
	#use the interpolation to get the evenly spaced flux
	fluxslice = wl_interp(wlslice)
	#return the new wavelength and flux
	return np.array(wlslice), np.array(fluxslice)

def norm_spec(wl, model, data):
	frac = data/model
	p_fitted = np.polynomial.Polynomial.fit(wl, frac, deg=2)
	return data/p_fitted(wl)

def interp_2_spec(spec1, spec2, ep1, ep2, val):	
	ret_arr = []
	#make sure the two spectra are the same length
	if len(spec1) == len(spec2):
		#go through the spectra
		#the new value is the first gridpoint plus the difference between them weighted by the spacing between the two gridpoints and the desired value.
		#this is a simple linear interpolation at each wavelength point
		ret_arr = ((np.array(spec2) - np.array(spec1))/(ep2 - ep1)) * (val - ep1) + np.array(spec1)
		#return the new interpolated flux array
		return ret_arr

	#otherwise yell at me because i'm trying to interpolate things that don't have the same length
	else:
		return('the spectra must have the same length')

def make_varied_param(init, sig):
	#initialize the variance list to return
	var = []
	#loop through the std devs 
	for s in sig:
		#check to make sure that the std devs are physical, and if they aren't print them out
		if any(a < 0 for a in s):
			print(s, np.where(sig == s), init[np.where(sig == s)])
	#then, loop through each of the input parameters
	for n in range(len(init)):
		#and if it's a single value just perturb it and put it into the array to return
		try:
			var.append(np.random.normal(init[n], sig[n]))
		#if the input parameter is itself an array, perturb each value in the array with the appropriate std dev and then save it
		except:
			var.append(list(np.random.normal(init[n], sig[n])))
	#return the perturbed values
	return var

def find_model(temp, logg, metal, models = 'btsettl'):
	"""Finds a filename for a phoenix model with values that fall on a grid point.
	Assumes that model files are in a subdirectory of the working directory, with that subdirectory called "SPECTRA"
	and that the file names take the form "lte{temp}-{log g}-{metallicity}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
	The file should contain a flux column, where the flux is in units of log(erg/s/cm^2/cm/surface area). There should also be a
	wavelength file in the spectra directory called WAVE_PHOENIX-ACES-AGSS-COND-2011.fits, with wavelength in Angstroms.
	"""
	#if using the hires phoenix models call using the correct formatting
	if models == 'hires':
		temp = str(int(temp)).zfill(5)
		metal = str(float(metal)).zfill(3)
		logg = str(float(logg)).zfill(3)
		file = glob('SPECTRA/lte{}-{}0-{}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits.txt'.format(temp, logg, metal))[0]
		return file
	#or if using BT-Settl, the other supported model, do the same
	#but here assume that we want the metallicity = 0 scenario (since those are the only ones I have downloaded)
	elif models == 'btsettl':
		# temp = str(temp/1e2).zfill(5)
		temp = str(int(temp/1e2)).zfill(3)
		metal = 0.0
		logg = str(logg)
		file = glob('BT-Settl_M-0.0a+0.0/lte{}-{}-0.0a+0.0.BT-Settl.spec.7.txt'.format(temp, logg))[0]
		# file = glob('BT-Settl_M-0.5_a+0.2/lte{}-{}-0.5a+0.2.BT-Settl.spec.7.txt'.format(temp, logg))[0]
		return file

def spec_interpolator(w, trange, lgrange, specrange, npix = 3, resolution = 10000, metal = 0, write_file = True, models = 'btsettl'):
	'''Runs before emcee, to read in files to memory
	'''
	#first, read in the wavelength vector
	if models == 'hires':
		with open('SPECTRA/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits.txt', 'r') as wave:
			spwave = []
			for line in wave:
				spwave.append(float(line))
			wave.close()
			spwave = np.array(spwave)
		#and find the indices where it's within the desired wavelength range - because this will be SET
		idx = np.where((spwave >= min(specrange)) & (spwave <= max(specrange)))
		spwave = spwave[idx]

		#now we need to collect all the files within the correct temperature and log(g) range
		#get the files
		files = glob('SPECTRA/lte*txt')
		#initialize temperature and log(g) arrays
		t = []
		l = []
		#sort through and pick out the temperature and log(g) value from each file name
		for n in range(len(files)):
			nu = files[n].split('-')[0].split('e')[1]
			mu = float(files[n].split('-')[1])

			if len(nu) < 4:
				nu = int(nu) * 1e2
			else:
				nu = int(nu)

			#add the teff and log(g) values to the relevant arrays if they are 1) not redundant and 2) within the desired range
			if nu not in t and nu >= min(trange) and nu <= max(trange):
				t.append(nu)
			if mu not in l and mu >= min(lgrange) and mu <= max(lgrange):
				l.append(mu)
		#now, make a dictionary where each name is formatted as 'teff, log(g)' and the entry is the spectrum
		specs = {}

		#for each (temperature, log(g)) combination, we need to read in the spectrum
		#select out the correct wavelength region
		#and downsample
		for n in range(len(t)):
			for k in range(len(l)):
				print(n, k)
				#find the correct file
				file = find_model(t[n], l[k], metal, models = models)
				#read it in
				with open(file, 'r') as f1:
					spec1 = []
					for line in f1:
						spec1.append(float(line))
				#select the right wavelength region
				spec1 = np.array(spec1)[idx]

				#downsample - default is 3 pixels per resolution element
				res_element = np.mean(spwave)/resolution
				spec_spacing = spwave[1] - spwave[0]
				if npix * spec_spacing < res_element:
					factor = (res_element/spec_spacing)/npix
					wl, spec1 = redres(spwave, spec1, factor)

				#next, we just add it to the dictionary with the correct (temp, log(g)) tuple identifying it

				specs['{}, {}'.format(t[n], l[k])] = np.array(spec1)

		specs['wl'] = np.array(wl)

	if models == 'btsettl':
		files = glob('BT-Settl_M-0.0a+0.0/lte*')

		#initialize temperature and log(g) arrays
		t = []
		l = []
		#sort through and pick out the temperature and log(g) value from each file name
		for n in range(len(files)):
			nu = files[n].split('-')[2].split('e')[1]
			mu = float(files[n].split('-')[3])

			nu = int(float(nu) * 1e2)

			#add the teff and log(g) values to the relevant arrays if they are 1) not redundant and 2) within the desired range
			if nu not in t and nu >= min(trange) and nu <= max(trange):
				t.append(nu)
			if mu not in l and mu >= min(lgrange) and mu <= max(lgrange):
				l.append(mu)
		#now, make a dictionary where each name is formatted as 'teff, log(g)' and the entry is the spectrum
		specs = {}; wls = {}
		wl = np.arange(min(specrange), max(specrange), 0.2)
		#for each (temperature, log(g)) combination, we need to read in the spectrum
		#select out the correct wavelength region
		#and downsample
		for n in range(len(t)):
			for k in range(len(l)):
				print(n, k)
				#find the correct file and read it in
				spold, spec1 = [], []
				#then find the correct file and save all the values that fall within the requested spectral range
				with open(find_model(t[n], l[k], metal, models = 'btsettl')) as file:
					for line in file:
						li = line.split(' ')
						if float(li[0]) >= min(specrange) - 100 and float(li[0]) <= max(specrange) + 100:
							spold.append(float(li[0])); spec1.append(float(li[1]))

				# spold, spec1 = np.genfromtxt(find_model(t[n], l[k], metal, models = 'btsettl')).T
				# spold, spec1 = spold[np.where((spold >= min(specrange) - 100) & (spold <= max(specrange) + 100))], spec1[np.where((spold >= min(specrange) - 100) & (spold <= max(specrange) + 100))]
				
				#next, we just add it to the dictionary with the correct (temp, log(g)) tuple identifying it
				specs['{}, {}'.format(t[n], l[k])] = np.array(spec1)
				wls['{}, {}'.format(t[n], l[k])] = np.array(spold)

		#now, for each spectrum we need to impose instrumental broadening
		for k in specs.keys():
			#select a spectrum and create an interpolator for it
			itep = interp1d(wls[k], specs[k])
			#then interpolate it onto the correct wavelength vcector
			spflux = itep(wl)
			#then instrumentally broaden the full spectrum
			ww, brd = broaden(wl[np.where((wl >= min(w))&(wl <= max(w)))], spflux[np.where((wl >= min(w))&(wl <= max(w)))], resolution)
			#we want to save the spectrum at original resolution where we can for better photometry
			#so create a new spectrum at original resolution outside the data spectrum range and at the data resolution inside it 
			newsp = np.concatenate((spflux[np.where(wl < min(w))], brd, spflux[np.where(wl > max(w))]))
			#and save to the dictionary
			specs[k] = newsp

		#fix the wavelength vector so it matches the spectrum wavelength scale
		wlnew = np.concatenate((wl[np.where(wl<min(w))], ww, wl[np.where(wl>max(w))]))
		#save the wavelength vector since it's now a common vector amongst all the spectra
		specs['wl'] = np.array(wlnew)
	#return the spectrum dictionary
	return specs

def get_spec(temp, log_g, reg, specdict, metallicity = 0, normalize = False, wlunit = 'aa', plot = False, models = 'btsettl', resolution = 1000, reduce_res = False, npix = 3):
	"""Creates a spectrum from given parameters.
	
	TO DO: add a path variable so that this is more flexible, add contingency in the homemade interpolation for if metallicity is not zero
	"""
	#we have to:
	#read in the synthetic spectra
	#pick our temperature and log g values (assume metallicity is constant for now)
	#pull a spectrum 
	#initialize a time variable if i want to check how long this takes to run
	time1 = time.time()
	#list all the spectrum files
	if models == 'hires':
		files = glob('SPECTRA/lte*txt')

		#initialize a tempeature array
		t = []
		#sort through and pick out the temperature value from each file name
		for n in range(len(files)):
			nu = files[n].split('-')[0].split('e')[1]
			if len(nu) < 4:
				nu = int(nu) * 1e2
				t.append(nu)
			else:
				t.append(int(nu))
		#sort the temperature array so it's in order
		t = sorted(t)
		#initialize a non-redundant array
		temps = [min(t)]

		#go through the sorted array and if the temperature isn't already in the non-redundant array, put it in
		for n, tt in enumerate(t):
			if tt > temps[-1]:
				temps.append(tt)

	if models == 'btsettl':
		files = glob('BT-Settl_M-0.0a+0.0/lte*')
		#initialize temperature and log(g) arrays
		t = []
		l = []
		#sort through and pick out the temperature and log(g) value from each file name
		for n in range(len(files)):
			nu = files[n].split('-')[2].split('e')[1]
			nu = int(float(nu) * 1e2)

			#add the teff and log(g) values to the relevant arrays if they are 1) not redundant and 2) within the desired range
			if nu not in t:
				t.append(nu)

			temps = sorted(t)

	#find the closest temperature to the input value
	t1_idx = find_nearest(temps, temp)

	#if that input value is on a grid point, make the second spectrum the same temperature
	if temps[t1_idx] == temp:
		t2_idx = t1_idx
	#if the nearest temp value is above the input value, the other temperature should fall below
	elif temps[t1_idx] > temp:
		t2_idx = t1_idx - 1
	#otherwise the second temperature should fall above
	else:
		t2_idx = t1_idx + 1

	#temp1 and temp2 have been selected to enclose the temperature, or to be the temperature exactly if the temp requested is on a grid point
	temp1 = temps[t1_idx]
	temp2 = temps[t2_idx]

	#now do the same thing for log(g)
	if models == 'btsettl':
		l = sorted([float(files[n].split('-')[3]) for n in range(len(files))])
	if models == 'hires':
		l = sorted([float(files[n].split('-')[1]) for n in range(len(files))])

	lgs = [min(l)]

	for n, tt in enumerate(l):
		if tt > lgs[-1]:
			lgs.append(tt)

	lg1_idx = find_nearest(lgs, log_g)
	 
	if lgs[lg1_idx] == log_g:
		lg2_idx = lg1_idx
	elif lgs[lg1_idx] > log_g:
		lg2_idx = lg1_idx - 1
	else:
		lg2_idx = lg1_idx + 1

	lg1 = lgs[lg1_idx]
	lg2 = lgs[lg2_idx]

	#so now I have four grid points: t1 lg1, t1 lg2, t2 lg1, t2 lg2. now i have to sort out whether some of those grid points are the same value
	#first, get the wavelength vector for everything 
	spwave = specdict['wl']
	#define a first spectrum using t1 lg1 and unpack the spectrum 

	#if I'm looking for a log(g) and a temperature that fall on a grid point, things are easy
	#just open the file 

	#the hires spectra are in units of erg/s/cm^2/cm, so divide by 1e8 to get erg/s/cm^2/A
	if lg1 == lg2 and temp1 == temp2:
		spflux = specdict['{}, {}'.format(temp1, lg1)]
		if models == 'hires':
			spflux /= 1e8

	#If the points don't all fall on the grid points, we need to get the second spectrum at point t2 lg2, as well as the cross products
	#(t1 lg2, t2 lg1)
	else:
		#find all the spectra 
		spec1 = specdict['{}, {}'.format(temp1, lg1)]
		spec2 = specdict['{}, {}'.format(temp2, lg2)]
		t1_inter = specdict['{}, {}'.format(temp1, lg2)]
		t2_inter = specdict['{}, {}'.format(temp2, lg1)]

		#if using hires correct the models to get the correct spectrum values
		if models == 'hires':
			spec1, spec2, t1_inter, t2_inter = spec1/1e8, spec2/1e8, t1_inter/1e8, t2_inter/1e8

		#if t1 and t2 AND lg1 and lg2 are different, we need to interpolate first between the two log(g) points, then the two teff points
		if lg1 != lg2 and temp1 != temp2:
			t1_lg = interp_2_spec(spec1, t1_inter, lg1, lg2, log_g)
			t2_lg = interp_2_spec(t2_inter, spec2, lg1, lg2, log_g)

			tlg = interp_2_spec(t1_lg, t2_lg, temp1, temp2, temp)

		#or if we're looking at the same log(g), we only need to interpolate in temperature
		elif lg1 == lg2 and temp1 != temp2:
			tlg = interp_2_spec(spec1, spec2, temp1, temp2, temp)

		#similarly, if we're using the same temperature but different log(g), we only interpolate in log(g)
		elif temp1 == temp2 and lg1 != lg2:
			tlg = interp_2_spec(spec1, spec2, lg1, lg2, log_g)
		#if you want, make a plot of all the different spectra to compare them
		#this only plots the final interpolated spectrum and the two teff points that are interpolated, after the log(g) interpolation has occurred
		if plot == True:
			wl1a, tla = make_reg(spwave, tlg, [1e4, 1e5])
			wl1a, t1l1a = make_reg(spwave, t1_lg, [1e4, 1e5])
			wl1a, t1l2a = make_reg(spwave, t2_lg, [1e4, 1e5])
			plt.loglog(wl1a, tla, label = 'tl')
			plt.loglog(wl1a, t1l1a, label = 't1l1')
			plt.loglog(wl1a, t1l2a, label = 't1l2')
			plt.legend()
			plt.show()
		
		#reassign some variables used above to match with the environment outside the if/else statement
		# spwave = spwave
		spflux = tlg

	#convert the requested region into angstroms to match the wavelength vector
	reg = np.array(reg)*1e4

	# #make sure the flux array is a float not a string
	# spflux = [float(s) for s in spflux]
	#and truncate the wavelength and flux vectors to contain only the requested region
	spwave, spflux = spwave[np.where((spwave >= min(reg)) & (spwave <= max(reg)))], spflux[np.where((spwave >= min(reg)) & (spwave <= max(reg)))] #make_reg(spwave, spflux, reg)
	#you can choose to normalize to a maximum of one
	if normalize == True:
		spflux /= max(spflux)

	#this is the second time object in case you want to check runtime
	# print('runtime for spectral retrieval (s): ', time.time() - time1)
	#broaden the spectrum to mimic the dispersion of a spectrograph using the input resolution
	# spwave, spflux = broaden(spwave, spflux, resolution)

	#depending on the requested return wavelength unit, do that, then return wavelength and flux as a tuple
	if wlunit == 'aa': #return in angstroms
		return np.array(spwave), np.array(spflux)
	elif wlunit == 'um':
		spwave = spwave * 1e-4
		return np.array(spwave), np.array(spflux)
	else:
		factor = float(input('That unit is not recognized for the return unit. \
			Please enter a multiplicative conversion factor to angstroms from your unit. For example, to convert to microns you would enter 1e-4.'))
		spwave = [s * factor for s in spwave]

		return np.array(spwave), np.array(spflux)

def get_transmission(f, res):
	'''retrieve a transmission curve at a given reoslution. Options are hardcoded because lots of the filter files have different structures.
	'''
	#first, we have to figure out what filter this is
	#make sure it's lowercase
	f = f.lower().strip(',')
	#get the system and filter from the input string
	try:
		if ',' in f:
			syst, fil = f.split(','); syst = syst.strip(); fil = fil.strip()
		#be lazy and hard-code some of these because it's easier and there's a limited number of options in the Furlan paper
		#and everything has a slightly different file format because of course this should be as hard as possible
		else:
			fil = f
			if fil in 'i':
				syst = 'cousins'
			elif fil in 'ubvr':
				syst = 'johnson'
			elif fil == 'kp' or fil == 'kshort':
				syst = 'keck'
			elif fil in 'jhks':
				syst = '2mass'
			elif fil in '562 692 880':
				syst = 'dssi'
			elif fil in 'kepler':
				syst = 'kep'
			elif fil in 'brgamma':
				syst = 'nirc2'
	except:
		print('Please format your filter as, e.g., "Johnson, V". The input is case insensitive.')
	#now get the fits file version of the transmission curve from the "bps" directory
	#which should be in the same directory as the code
	#many of these have their own weird format so hard code some of them 
	if fil == 'lp600' or fil == 'LP600': #got the transmission curve from Baranec et al 2014 (ApJL: High-efficiency Autonomous Laser Adaptive Optics)
		filtfile = np.genfromtxt('bps/lp600.csv', delimiter = ',')
		t_wl, t_cv = filtfile[:,0]* 10, filtfile[:,1]
	elif syst.lower() == 'gaia' and fil == 'g':
		t_wl, t_cv = np.genfromtxt('bps/gaia_g_pb.txt').T
		t_wl *= 1e1
	elif syst.lower() == 'gaia' and fil == 'rp':
		t_wl, t_cv = np.genfromtxt('bps/gaia_rp_pb.txt').T
		t_wl *= 1e1
	elif syst.lower() == 'gaia' and fil == 'bp':
		t_wl, t_cv = np.genfromtxt('bps/gaia_bp_pb.txt').T
		t_wl *= 1e1
	elif syst == 'kep':
		t_wl, t_cv = np.genfromtxt('bps/Kepler_Kepler.K.dat').T
	elif syst == 'nirc2':
		t_wl, t_cv = np.genfromtxt('bps/Keck_NIRC2.Brgamma.dat').T
	elif syst == '2mass' or syst == '2MASS':  
		if fil.strip() == 'j' or fil.strip() == 'h':
			filtfile = fits.open('bps/2mass_{}_001_syn.fits'.format(fil.strip()))[1].data
			t_wl, t_cv = filtfile['WAVELENGTH'], filtfile['THROUGHPUT']
		if fil.strip() == 'k' or fil.strip() == 'ks':
			filtfile = np.genfromtxt('bps/2MASS_2MASS.Ks.dat')
			t_wl, t_cv = filtfile[:,0], filtfile[:,1]/max(filtfile[:,1])
	elif syst == 'dssi': #from the SVO transmission curve database for DSSI at Gemini North
		filtfile = np.genfromtxt('bps/DSSI_{}nm.dat'.format(fil))
		t_wl = filtfile[:,0]; t_cv = filtfile[:,1]
	elif syst == 'sdss':
		t_wl, t_cv = np.genfromtxt('bps/SLOAN_SDSS.{}prime_filter.dat'.format(fil)).T
	elif syst == 'sloan':
		fname = np.array(['u\'', 'g\'', 'r\'', 'i\'', 'z\''])
		n = np.where(fil + '\'' == fname)[0][0]
		filtfile = Table(fits.open('bps/sdss.fits')[n+1].data)
		t_wl = np.array(filtfile['wavelength']); t_cv = np.array(filtfile['respt'])	
	elif syst == 'keck' and fil == 'kp': #taken from the keck website I think? or from SVO
		filtfile = np.genfromtxt('bps/keck_kp.txt')
		t_wl, t_cv = filtfile[:,0], filtfile[:,1]	
		t_wl *= 1e4
	elif syst == 'keck' and fil == 'kshort':
		t_wl, t_cv = np.genfromtxt('bps/Keck_NIRC2.Ks.dat').T

	else:
		filtfile = fits.open('bps/{}_{}_002.fits'.format(syst, fil))[1].data

		t_wl, t_cv = filtfile['WAVELENGTH'], filtfile['THROUGHPUT']

	#calculate the size of the mean resolution element and then the number of total resolution elements 
	#this was for way back when I was down-weighting the spectrum by this value, but I don't do that anymore
	#so I suppose this is deprecated now, but it's still built in as a nuisance parameter for all the calls to this function so might as well leave it in for now
	res_element = np.mean(t_wl)/res
	n_resel = (max(t_wl) - min(t_wl))/res_element
	#return the wavelength array, the transmission curve, the number of resolution elements in the bandpass, and the central wavelength
	return t_wl, t_cv, n_resel, np.mean(t_wl)

def make_composite(teff, logg, rad, distance, contrast_filt, phot_filt, r, specs, ctm, ptm, tmi, tma, vs, nspec = 2, normalize = False, res = 1000, npix = 3, models = 'btsettl', plot = False):
	"""add spectra together given an array of spectra and flux ratios. Returns a spectrum, composite photometry, and contrasts. Can return component data if plot = True.
	"""

	#unpack the contrast and photometry lists
	#wls and tras are the wavelength and transmission arrays for the contrast list; n_res_el is the number of resolution elements in each filter, and 
	#cwl is the central wavelength for each contrast filter
	wls, tras, n_res_el, cwl = ctm
	#same order for these four variables but for the photometry, not the contrasts
	phot_wls, phot_tras, phot_resel, phot_cwl = ptm

	#now find the wavelength global extrema by checking the min and max of each transmission curve wavelength vector
	wlmin, wlmax = np.inf, 0
	for w in wls:
		if min(w) < wlmin:
			wlmin = min(w)
		if max(w) > wlmax:
			wlmax = max(w)
	for p in phot_wls:
		if min(p) < wlmin:
			wlmin = min(p)
		if max(p) > wlmax:
			wlmax = max(p)

	#the "plot" keyword doesn't plot anything, it's here as a flag for when the function call is for plotting
	#as things are set up I have it to also calculate the kepler magnitude when called with "plot = True" because when creating diagnostic plots I use it for other analysis
	if plot == True:
		ran, tm, a, b = get_transmission('kepler', res)
		if min(ran) < wlmin:
			wlmin = min(ran)
		if max(ran) > wlmax:
			wlmax = max(ran)

	#get the primary star wavelength array and spectrum 
	#the min and max wavelength points will be in Angstroms so we need to make them microns for the function call
	#returns in erg/s/cm^2/A/surface area
	pri_wl, pri_spec = get_spec(teff[0], logg[0], [min(min(r), tmi/1e4, wlmin/1e4) - 1e-4, max(max(r), tma/1e4, wlmax/1e4) + 1e-4], specs, normalize = False, resolution = res, npix = npix, models = models)
	#convert spectrum to a recieved flux at earth surface: multiply by surface area (4pi r^2) to get the luminosity, then divide by distance (4pi d^2) to get flux
	if not type(distance) == bool: #if we're fitting for distance convert to a flux
		di = 1/distance
		pri_spec *= (rad[0]*6.957e+10/(di * 3.086e18))**2

	#now we need to get the secondary (and possibly higher-order multiple) spectra
	#given the way the spectral retrieval code works, as long as the wavelength range is the same the spectra will be on the same grid
	#so I'm just going to stack them with the primary wavelength and spectrum - going to save the wavelength just in case I need to troubleshoot later
	for n in range(1, len(teff)):
		sec_wl, sec_spec = get_spec(teff[n], logg[n], [min(min(r), tmi/1e4, wlmin/1e4) - 1e-4, max(max(r), tma/1e4, wlmax/1e4) + 1e-4], specs, normalize = False, resolution = res, npix = npix, models = models)
		if not type(distance) == bool: #if fitting for distance convert to flux
			di = 1/distance
			sec_spec = sec_spec * (rad[0]*rad[n]*6.957e+10/(di * 3.086e18))**2 
		else: #otherwise need to alter the flux based on the radius ratio
			#we've set the primary radius to 1, so this radius is just the square of the radius ratio in cm
			sec_spec = sec_spec * rad[n-1]**2

		#save the secondary wavelength and spectrum in the total system arrays 
		#this is to provide flexibility if I want to fit for a triple intead of a binary
		pri_wl = np.row_stack((pri_wl, sec_wl)); pri_spec = np.row_stack((pri_spec, sec_spec))

	"""
	calculate contrast magnitudes
	"""
	#define an array to hold all my "instrumental" fluxes
	mag = np.zeros((len(contrast_filt), len(teff)))


	#loop through each filter 
	for n, f in enumerate(contrast_filt):
		#get the wavelength range and transmission curve
		ran, tm = wls[n], tras[n]
		#pick out the region of stellar spectrum that matches the curve
		w = pri_wl[0, :][np.where((pri_wl[0,:] <= max(ran)) & (pri_wl[0,:] >= min(ran)))]
		#and interpolate the transmission curve so that it matches the stellar spectral resolution
		intep = interp1d(ran, tm)
		tran = intep(w)
		#now for each star
		for k in range(len(teff)):
			#pick out the spectrum over the appropriate region
			s = pri_spec[k][np.where((pri_wl[0,:] <= max(ran)) & (pri_wl[0,:] >= min(ran)))]
			t_spec = [s[p] * tran[p] for p in range(len(s))]
			#put it through the filter by multiplying by the transmission, then integrate to finally get the instrumental flux
			m = np.trapz(t_spec, w)
			#and add it to the array in the appropriate place for this star and filter
			mag[n][k] = -2.5 * np.log10(m)
	#now we have a set of synthetic magnitudes for each filter with a known flux ratio 
	#don't need to apply filter zp etc because all that matters is the differential measurement
	#now we need to calculate the contrast
	#i'm only going to set up the contrast calculation for a binary right now
	#need to just take the flux ratio from flux (secondary/primary)

	if float(nspec) == 2:
		contrast = [mag[n][1] - mag[n][0] for n in range(len(contrast_filt))]

		#make the composite spectrum
		spec1 = pri_spec[0,:] + pri_spec[1,:]

	else:
		contrast1 = [mag[n][1] - mag[n][0] for n in range(len(contrast_filt))]
		contrast2 = [mag[n][2] - mag[n][0] for n in range(len(contrast_filt))]
		contrast = np.concatenate((contrast1[:int(len(contrast_filt)/2)], contrast2[int(len(contrast_filt)/2):]))

		spec1 = pri_spec[0,:] + pri_spec[1,:] + pri_spec[2,:]

	#pri_wl[0] and spec1 are the composite wavelength and spectrum vectors
	#now we have to go through each filter in phot_filt and do the photometry using phot_wls and phot_tras
	phot_phot = []
	#2mass zero points (values from Cohen+ 2003); sdss filter values from SVO filter profile service
	#[r, i, z, J, H, Ks]
	zp_jy = [3112.91, 2502.62, 1820.98, 1594, 1024, 666.7] #zero point in Jy (10^-23 erg/s/cm^2/Hz)
	cw = [6246.98, 7718.28, 10829.83, 1.235e4, 1.662e4, 2.159e4] #A
	bp_width = [1253.71, 1478.93, 4306.72, 1620, 2509, 2618]
	zp = [zp_jy[n]*bp_width[n]/(3.336e4 * cw[n]**2) for n in range(len(zp_jy))] #convert to a zero point in flux
	#conversion from ab mag to vega mag for the SDSS filters 
	mab_minus_mvega = [0.16,0.37,0.54] #to get m_vega, mab - mvega = 0.08 -> mvega = mab - 0.08

	#if only using 2MASS mags truncate the zero point array
	if len(phot_filt) == 3:
		fs = ['2MASS_J', '2MASS_H', '2MASS_Ks']
	else:
		fs = ['SDSS_r', 'SDSS_i', 'SDSS_z', '2MASS_J', '2MASS_H', '2MASS_Ks']

	for n in range(len(phot_filt)):
		# get the internal default library of passbands filters
		f = lib[fs[n]] #['SDSS_g']#pyphot.Filter(phot_wls[n], bp, dtype = 'photon', unit = 'Angstrom')
		# compute the integrated flux through the filter f
		# note that it work on many spectra at once
		fluxes = f.get_flux(pri_wl[0,:]*pyphot.unit('AA'), spec1*pyphot.unit('erg/s/cm**2/AA'))

		if '2MASS' in fs[n]:
			# convert to vega magnitudes
			mags = -2.5 * np.log10((fluxes/f.Vega_zero_flux).value)
		else:
			mags = -2.5 * np.log10((fluxes/f.AB_zero_flux).value)	
		phot_phot.append(mags)

	#if I'm calling this during the plot function, I also want to get the kepler photometry for each component
	if plot == True:
		ran, tm, a, b = get_transmission('kepler', res)
		gaia_ran, gaia_tm, gaia_a, gaia_b = get_transmission('gaia,g', res)

		#interpolate the transmission curve to the data wavelength scale
		intep = interp1d(ran, tm); data_tm = intep(pri_wl[0,:][np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))])

		gaia_intep = interp1d(gaia_ran, gaia_tm); gaia_data_tm = gaia_intep(pri_wl[0,:][np.where((pri_wl[0,:] >= min(gaia_ran)) & (pri_wl[0,:] <= max(gaia_ran)))])

		if nspec == 2:
			#calculate the photometry for each component by convolving with the tm curve, integrating, and dividing by the zero point
			pri_phot = np.trapz(np.array(pri_spec[0,:])[np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))] * data_tm, pri_wl[0,:][np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))])
			sec_phot = np.trapz(np.array(sec_spec)[np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))] * data_tm, pri_wl[0,:][np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))])

			#convert to vega mag from flux values
			pri_vegamag = -2.5*np.log10(pri_phot); sec_vegamag = -2.5*np.log10(sec_phot)

			 
			#calculate the photometry for each component by convolving with the tm curve, integrating, and dividing by the zero point
			# gaiafilt = pyphot.Filter(gaia_ran*pyphot.unit['AA'], gaia_tm)

			#convert to vega mag from flux values
			# gaia_pri_vegamag = -2.5*np.log10(gaiafilt.get_flux(pri_wl[0,:], pri_spec[0,:]*pyphot.unit('erg/s/cm**2/AA'))) -  25.6873668671; gaia_sec_vegamag = -2.5*np.log10(gaiafilt.get_flux(pri_wl[0,:], sec_spec*pyphot.unit('erg/s/cm**2/AA'))) -  25.6873668671
			# print(gaia_pri_vegamag)
			f = lib['Gaia_G']
			gaiamag = -2.5 * np.log10((f.get_flux(pri_wl[0,:]*pyphot.unit('AA'), spec1*pyphot.unit('erg/s/cm**2/AA'))/f.Vega_zero_flux).value)
			gaia_pri_vegamag = -2.5 * np.log10((f.get_flux(pri_wl[0,:]*pyphot.unit('AA'), pri_spec[0,:]*pyphot.unit('erg/s/cm**2/AA'))).value) - f.Vega_zero_mag
			gaia_sec_vegamag = -2.5 * np.log10((f.get_flux(pri_wl[0,:]*pyphot.unit('AA'), sec_spec*pyphot.unit('erg/s/cm**2/AA'))).value) - f.Vega_zero_mag
			#return the wavelength array, the composite spectrum, the component spectra, and the component kepler magnitudes
			return np.array(pri_wl[0,:]), np.array(spec1), np.array(pri_spec[0,:]), np.array(sec_spec), pri_vegamag, sec_vegamag, gaia_pri_vegamag, gaia_sec_vegamag, gaiamag

		if nspec ==3:
			#calculate the photometry for each component by convolving with the tm curve, integrating, and dividing by the zero point
			pri_phot = np.sum(np.array(pri_spec[0,:])[np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))] * data_tm)/zp
			sec_phot = np.sum(np.array(pri_spec[1,:])[np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))] * data_tm)/zp
			tri_phot = np.sum(np.array(pri_spec[2,:])[np.where((pri_wl[0,:] >= min(ran)) & (pri_wl[0,:] <= max(ran)))] * data_tm)/zp

			#convert to vega mag from flux values
			pri_vegamag = -2.5*np.log10(pri_phot); sec_vegamag = -2.5*np.log10(sec_phot); tri_vegamag = -2.5*np.log10(tri_phot)

			#return the wavelength array, the composite spectrum, the component spectra, and the component kepler magnitudes
			return np.array(pri_wl[0,:]), np.array(spec1), np.array(pri_spec[0,:]), np.array(pri_spec[1,:]), np.array(pri_spec[2,:]), pri_vegamag, sec_vegamag, tri_vegamag
	
	else:
		return np.array(pri_wl[0,:]), np.array(spec1), [c for c in contrast], np.array([float(p) for p in phot_cwl]), np.array([p for p in phot_phot])

def opt_prior(vals, pval, psig):
	"""Imposes a gaussian prior using a given prior value and standard deviation
	"""
	#initialize a list for likelihood values
	pp = []
	#for each value we're calculating the prior for
	if len(pval) == 1 or type(pval) == float:
		try:
			pp.append(((float(vals)-float(pval))/float(psig))**2)
		except:
			pp.append(((vals[0]-pval[0])/psig[0])**2)
	else:
		for k, p in enumerate(pval):
			#as long as the prior is defined
			if p != 0:
				#calculate the likelihood
				like = ((vals[k] - pval[k])/(psig[k]))**2
				#and save it
				pp.append(like)

	#return the sum of the likelihoods
	return np.sum(pp)

def fit_spec(n_walkers, dirname, wl, flux, err, reg, t_guess, av, rad_guess, fr_guess, specs, tlim, distance, ctm, ptm, tmi, tma, vs, matrix, ra, dec, nspec = 2, steps = 200, models = 'btsettl', dist_fit = True, rad_prior = False):
	"""Performs a modified gibbs sampler MCMC using a reduced chi-square statistic.
	"""
	#make sure wl is in Angstroms 
	wl *= 1e4

	#unpack the distance as long as I'm using it 
	if not type(distance) == bool:
		dist, pprior, psig = distance

	#unpack the initial extinction guess (although I currently don't fit for Av)
	extinct_guess, eprior, esig = av

	#note that fr_guess[0] = contrasts, fr_guess[1] = contrast errors, [2] = filters, [3] = unres phot values, [4] = errors, [5] = filters
	#phot is in flux, contrast is in mags
	lg_guess = [get_logg(t, matrix) for t in t_guess]
	#make an initial guess spectrum, contrast, and photometry using absolute radii if using a distance and using a radius ratio otherwis
	if not type(distance) == bool:
		wave1, init_cspec, contrast, phot_cwl, phot = make_composite(t_guess, lg_guess, rad_guess, dist, fr_guess[2], fr_guess[5], reg, specs, ctm, ptm, tmi, tma, vs, models = models, nspec = nspec)

	else:
		wave1, init_cspec, contrast, phot_cwl, phot = make_composite(t_guess, lg_guess, rad_guess, distance, fr_guess[2], fr_guess[5], reg, specs, ctm, ptm, tmi, tma, vs, models = models, nspec = nspec)

	#extinct the initial spectrum and photometry guesses
	# init_cspec = extinct(wave1, init_cspec, extinct_guess)
	init_phot = -2.5*np.log10(extinct(phot_cwl, 10**(-0.4*phot), extinct_guess))

	#interpolate the model onto the data wavelength vector
	# intep = interp1d(wave1, init_cspec)
	# init_cspec = intep(wl)

	#normalize the model to match the median of the data
	# init_cspec*=np.median(flux)/np.median(init_cspec)
	# flux = norm_spec(wl, init_cspec, flux) 

	#calculate the chi square value of the spectrum fit
	# ic = chisq(init_cspec, flux, err)
	# iic = np.sum(ic)/len(ic)*3

	#calculate the chi square for the contrast fit
	chi_contrast = chisq(contrast, fr_guess[0], fr_guess[1])
	icontrast = np.sum(chi_contrast)

	#if using any distance calculate the photometry chi square and then the total chi square after weighting the spectrum chi square appropriately
	if not type(distance) == bool:
		ip = chisq(phot, fr_guess[3], fr_guess[4])
		iphot = np.sum(ip)
		# print(iic, icontrast, iphot)
		init_cs = np.sum((icontrast, iphot))
	#otherwise calculate the total chi square as the sum of the contrast and the spectrum chi squares after weighting the spectrum chi^2 appropriately
	else:
		init_cs = np.sum((icontrast))
	#if the distance is from Gaia, impose a distance prior
	if not type(distance) == bool and dist_fit == True:
		init_cs += opt_prior([dist], [pprior], [psig])

	if rad_prior == True:
		model_radius1 = get_radius(t_guess[0], matrix)
		model_radius2 = get_radius(t_guess[1], matrix)
		#assume sigma is 5%, which is a pretty typical radius measurement uncertainty
		# print(rad_guess)
		if nspec == 2:
			init_cs += opt_prior(rad_guess, [model_radius1, model_radius2/model_radius1], [0.05*r for r in rad_guess])

		elif nspec == 3:
			model_radius3 = get_radius(t_guess[2], matrix)
			init_cs += opt_prior(rad_guess, [model_radius1, model_radius2/model_radius1, model_radius3/model_radius1], [0.05*r for r in rad_guess])

	pos = SkyCoord(ra*u.deg, dec*u.deg, distance = (1/dist)*u.pc)
	av_guess = bayestar(pos, mode = 'samples') * 3.1 * 0.884
	av_guess_mu = np.mean(av_guess); av_guess_sig = np.std(av_guess)
	if av_guess_sig == 0:
		av_guess_sig = 0.05
	init_cs += opt_prior([extinct_guess], [av_guess_mu], [av_guess_sig])
	#that becomes your comparison chi square
	chi = init_cs
	#make a random seed
	#this is necessary because we initialized everything from the same node, so each core needs to generate its own seed to make sure that the random calls are independent
	r = np.random.RandomState()

	#savechi will hang on to the chi square value of each fit
	savechi = [init_cs]
	#savetest will hang on to the test values for each guess
	savetest = [t_guess, lg_guess, extinct_guess, rad_guess]

	"""
	This section is for if the fit will use any sort of distance measurement 
	"""
	if not type(distance) == bool:

		#sp will hang on to the tested set of parameters at the end of each iteration
		#so will eventually become 2D but for now it just holds the initial guess
		sp = [t_guess, extinct_guess, rad_guess, dist]

		#si is the std dev for the gaussian calls the function makes to vary the test parameters
		#initially this is very coarse so we explore the parameter space rapidly
		if nspec == 2:
			si = [[250, 250], [0.05], [0.1 * r for r in rad_guess], [0.02 * dist]]
		elif nspec == 3:
			si = [[250, 250, 250], [0.05], [0.1 * r for r in rad_guess], [0.05 * dist]]

		#gi is the guess for an individual function call so right now it's just the initial guess
		#this is what will vary at each function call and be saved in sp, which was initialized above
		gi = [t_guess, extinct_guess, rad_guess, dist]

		#initialize a step counter and a cutoff counter
		#the cutoff it to make sure that if the function goes bonkers it will still eventually end 
		n = 0
		total_n = 0
		#as long as both counters are below the set limits 
		while n < steps and total_n < (50 * steps):

			#if we're halfway through reduce the step size significantly to refine the fit in chi^2 surface assuming the coarse step got us to the approximate correct minimum
			if n > (steps/2):
				if nspec == 2:
					si = [[20, 20], [0.01], [0.05 * r for r in rad_guess], [0.005*dist]]
				elif nspec == 3:
					si = [[20, 20, 20], [0.01], [0.05 * r for r in rad_guess], [0.01*dist]]


			#and then vary all the parameters simultaneously using the correct std dev
			var_par = make_varied_param(gi, si)

			# print(var_par, tlim, llim)
			#make sure that everything that got varied was inside the parameter limits
			if all(min(tlim) < v < max(tlim) for v in var_par[0])  and 0 <= var_par[1] and 0.05 <= var_par[2][0] <= 1.5 and 0.05 < var_par[2][1] < 1\
				 and 1/10 > var_par[3] > 1/3000 and var_par[1] >= 0:

				while nspec == 3 and (var_par[2][2] >= var_par[2][1] or var_par[2][2] < 0):
					var_par[2][2] = var_par[2][1] * 0.9

				#we made it through, so increment the counters by 1 to count the function call
				total_n += 1
				n += 1

				pos = SkyCoord(ra*u.deg, dec*u.deg, distance = (1/var_par[3])*u.pc)
				av_guess = bayestar(pos, mode = 'samples') * 3.1 * 0.884
				av_guess_mu = np.mean(av_guess); av_guess_sig = np.std(av_guess)
				if av_guess_sig == 0:
					av_guess_sig = 0.05

				logg_guess = [get_logg(v, matrix) for v in var_par[0]]
				#create a test data set using the guess parameters
				test_wave1, test_cspec, test_contrast, test_phot_cwl, test_phot = make_composite(var_par[0], logg_guess, var_par[2], float(var_par[3]), fr_guess[2], fr_guess[5], reg, specs, ctm, ptm, tmi, tma, vs, models = models, nspec = nspec)

				#extinct the spectrum and photometry
				if var_par[1] > 0:
					# test_cspec = extinct(test_wave1, test_cspec, var_par[1])
					test_phot = -2.5*np.log10(extinct(test_phot_cwl, 10**(-0.4*test_phot), var_par[1]))

				#interpolate the test spectrum onto the data wavelength vector
				# intep = interp1d(test_wave1, test_cspec)
				# test_cspec = intep(wl)

				#normalize the test spectrum to match the data normalization
				# test_cspec*=np.median(flux)/np.median(test_cspec)

				#calculate the reduced chi square between data spectrum and guess spectrum 
				# tc = chisq(test_cspec, flux, err)
				# ttc = np.sum(tc)/len(tc) * 3

				#calculate the contrast chi square
				chi_contrast = chisq(test_contrast, fr_guess[0], fr_guess[1])
				tcontrast = np.sum(chi_contrast)

				#calculate the photometry chi square - this is a loop that is only for distance-provided inputs, so photometry will always be relevant
				chi_phot = chisq(test_phot, fr_guess[3], fr_guess[4])
				tphot = np.sum(chi_phot)

				# print(ttc, tcontrast, tphot, len(tc))

				#create the total chi square by summing the chi^2 after weighting the spectrum chi^2 appropriately
				test_cs = np.sum((tcontrast, tphot))

				test_cs += opt_prior(var_par[1], [av_guess_mu], [av_guess_sig])
				# print(1/var_par[4], var_par[2], av_guess_mu, av_guess_sig, opt_prior(var_par[2], [av_guess_mu], [av_guess_sig]), tcontrast, tphot, ttc*(len(chi_contrast) + len(chi_phot)))

				#if we're using a real distance instead of a fake distance, add in a distance prior to the guess 
				if dist_fit == True:
					test_cs += opt_prior([var_par[3]], [pprior], [psig])

				if rad_prior == True and nspec == 2:
					model_radius1 = get_radius(var_par[0][0], matrix)
					model_radius2 = get_radius(var_par[0][1], matrix)
					#assume sigma is 10% or 5%, which is a pretty typical radius measurement uncertainty
					# print(init_cs, opt_prior(var_par[3], [model_radius1, model_radius2/model_radius1], np.array(si[3])/2), [model_radius1, model_radius2/model_radius1], var_par[3], si[3])
					test_cs += opt_prior(var_par[2], [model_radius1, model_radius2/model_radius1], np.array(si[2]))

				elif rad_prior == True and nspec == 3:
					model_radius1 = get_radius(var_par[0][0], matrix)
					model_radius2 = get_radius(var_par[0][1], matrix)
					model_radius3 = get_radius(var_par[0][2], matrix)
					#assume sigma is 10% or 5%, which is a pretty typical radius measurement uncertainty
					# print(init_cs, opt_prior(var_par[3], [model_radius1, model_radius2/model_radius1], np.array(si[3])/2), [model_radius1, model_radius2/model_radius1], var_par[3], si[3])
					test_cs += opt_prior(var_par[2], [model_radius1, model_radius2/model_radius1, model_radius3/model_radius1], np.array(si[2]))

				#now, if the test chi^2 is better than the prevous chi^2
				if test_cs < chi:
					#replace the old best-fit parameters with the new ones
					gi = var_par
					#save the new chi^2
					chi = test_cs 
					#if we're more than halfway through, just go back to the small variations, don't start all over again
					if n > (steps/2):
						n = steps/2 + 1
					#but if we're less than halfway through, start the fit over because we want to go until we've tried n guesses without a better guess
					else:
						n = 0

				#save everything to the appropriate variables
				sp = np.vstack((sp, gi))
				savechi.append(chi)
				savetest.append(test_cs)

			#if any guess is outside of the limits, vary the offending parameter until it isn't anymore 
			else:
				#but still increment the total count by one, because these calls count too
				total_n += 1

				#temperatures
				while any(v < min(tlim) for v in var_par[0]) and total_n < (50*steps):
					total_n += 1
					var_par[0][np.where(var_par[0]<min(tlim))] += 100
				while any(v > max(tlim) for v in var_par[0]) and total_n < (50*steps):
					total_n += 1
					var_par[0][np.where(var_par[0] > max(tlim))] -= 100

				while var_par[0][0] < var_par[0][1] and total_n < (50*steps):
					total_n += 1
					var_par[0][1] -= 100

				#extinction
				while var_par[1] < 0 and total_n < (50*steps):
					total_n += 1
					var_par[1] += 0.1

				#radius
				while any(v < 0.05 for v in var_par[2]) and total_n < (50*steps):
					total_n += 1
					var_par[2][np.where(var_par[2]<0.05)] += 0.01

				#distance (as parallax)
				while var_par[3] > 1/100 and total_n < (50*steps):
					total_n += 1
					var_par[3] -= 0.01 * np.abs(var_par[3])
				while var_par[3] < 1/3000 and total_n < (50*steps):
					total_n += 1
					var_par[3] += 0.01* np.abs(var_par[3])

		if nspec == 2:
			#save all the guessed best-fit parameters to a file 
			f = open(dirname + '/params{}.txt'.format(n_walkers), 'a')
			for n in range(1, len(savechi)):
				f.write('{} {} {} {} {} {}\n'.format(sp[:][n][0][0], sp[:][n][0][1], float(sp[:][n][1]), sp[:][n][2][0], sp[:][n][2][1], float(sp[:][n][3])))
			f.close()
			#save all the best-fit chi^2 values to a file
			f = open(dirname + '/chisq{}.txt'.format(n_walkers), 'a')
			for n in range(1, len(savechi)):
				f.write('{} {}\n'.format(savechi[n], savetest[n]))
			f.close()

			#and then return the final best-fit value and chi^2 
			return '{} {} {} {} {} {}\n'.format(gi[0][0], gi[0][1], float(gi[1]), gi[2][0], gi[2][1], float(gi[3])), savechi[-1]
		elif nspec == 3:
			#save all the guessed best-fit parameters to a file 
			f = open(dirname + '/params{}.txt'.format(n_walkers), 'a')
			for n in range(len(savechi)):
				try:
					f.write('{} {} {} {} {} {} {} {}\n'.format(sp[:][n][0][0], sp[:][n][0][1], sp[:][n][0][2], float(sp[:][n][1]), sp[:][n][2][0], sp[:][n][2][1], sp[:][n][2][2], float(sp[:][n][3])))
				except:
					print('weird output? not sure what\'s going on here yet')
					f.write('{} {} {} {} {} {} {} {}\n'.format(sp[0][0], sp[0][1], sp[0][2], sp[1], sp[2][0], sp[2][1], sp[2][2], sp[3]))

			f.close()
			#save all the best-fit chi^2 values to a file
			f = open(dirname + '/chisq{}.txt'.format(n_walkers), 'a')
			for n in range(len(savechi)):
				f.write('{} {}\n'.format(savechi[n], savetest[n]))
			f.close()

			#and then return the final best-fit value and chi^2 
			return '{} {} {} {} {} {} {} {}\n'.format(gi[0][0], gi[0][1], gi[0][2], float(gi[1]), gi[2][0], gi[2][1], gi[2][2], float(gi[3])), savechi[-1]

def loglikelihood(p0, fr, nspec, ndust, data, err, broadening, r, specs, ctm, ptm, tmi, tma, vs, matrix, w = 'aa', dust = False, norm = True, mode = 'spec', av = True, optimize = False, models = 'btsettl'):
	#unpack data tuple into wavelength and data arrays
	wl, spec = np.array(data)

	#unpack the guess array using the number of spectra to fit to (hardcoded as two in the len call right now)
	#if distance is included in the guess unpack it
	if len(p0) == 6:
		t_guess, extinct_guess, rad_guess, dist_guess = p0[:nspec], p0[nspec], p0[nspec+1:2*nspec + 1], p0[2*nspec+1]
		#create the composite spectrum that corresponds to the guess values
		# t1 = time.time()
		lg_guess = [get_logg(t, matrix) for t in t_guess]
		wave1, init_cspec, contrast, phot_cwl, phot = make_composite(t_guess, lg_guess, rad_guess, dist_guess, fr[2], fr[5], r, specs, ctm, ptm, tmi, tma, vs, models = models, nspec = nspec)
		# print('time for composite: ', time.time() - t1)
	#otherwise just get the other terms 
	elif len(p0) == 8:
		t_guess, extinct_guess, rad_guess, dist_guess = p0[:nspec], p0[nspec], p0[nspec+1:2*nspec + 1], p0[2*nspec+1]
		#create the composite spectrum that corresponds to the guess values
		# t1 = time.time()
		lg_guess = [get_logg(t, matrix) for t in t_guess]
		wave1, init_cspec, contrast, phot_cwl, phot = make_composite(t_guess, lg_guess, rad_guess, dist_guess, fr[2], fr[5], r, specs, ctm, ptm, tmi, tma, vs, models = models, nspec = nspec)

	#if the extinction value is True extinct the spectrum and photometry using the guess value
	if av == True and extinct_guess > 0:
		# init_cspec = extinct(wave1, init_cspec, extinct_guess)
		init_phot = -2.5*np.log10(extinct(phot_cwl, 10**(-0.4*phot), extinct_guess))
	#otherwise just transfer the photometry to another variable for consistency and proceed without extincting anything
	else:
		init_phot = phot

	#interpolate the model onto the date wavelength scale
	# intep = interp1d(wave1, init_cspec)
	# init_cspec = intep(wl * 1e4)

	#normalize the model
	# init_cspec *= np.median(spec)/np.median(init_cspec)
	# spec = norm_spec(wl, init_cspec, spec) 


	#calculate the chi square value of that fit
	# ic = chisq(init_cspec, spec, err)
	# iic = np.sum(ic)/len(ic)

	#calculate the chi square for the contrast fit
	chi_contrast = chisq(contrast, fr[0], fr[1])
	icontrast = np.sum(chi_contrast)

	#if there is photometry involved (i.e., a distance)
	if len(p0) == 6 or len(p0) == 8:
		#calculate the chi square for the photometry
		chi_phot = chisq(init_phot, fr[3], fr[4])
		iphot = np.sum(chi_phot)
		#and the total chi square is the sum of the chi squares, where the spectrum is weighted to be as important as the combined photometry and contrasts
		init_cs = np.sum((icontrast, iphot))
	#if we're not using a distance at all, just weight the spectrum by the contrast chi square and then calculate the total chi square
	else:
		# iic *= (len(chi_contrast))
		init_cs = np.sum((icontrast))

	#if i'm running a simple optimization I just need to return a chi square
	if optimize == True:
		return init_cs
	#but usually I want to return the log likelihood for emcee 
	else:
		if np.isnan(init_cs):
			return -np.inf
		else:
			return -0.5 * init_cs

def logprior(p0, nspec, ndust, tmin, tmax, matrix, ra, dec, prior = 0, ext = True, dist_fit = True, rad_prior = False):
	#get the guess temperatures and surface gravities
	temps = p0[:nspec]
	#now there are a bunch of different situations and this is definitely not as pretty as it could be but I suppose it works fine
	#first, calculate if I'm using a distance 
	if len(p0) == 6 and dist_fit == True:
		#if i'm fitting for extinction, get Av, radii, and distance from the guess array
		if ext == True:
			extinct = p0[nspec]
			rad = p0[nspec + 1:2*nspec + 1]
			dist = p0[2*nspec + 1]
		#otherwise just get the radii and distance
		else:
			rad = p0[nspec:2*nspec]
			dist = p0[2*nspec + 1]

		pp = []

		#now check to make sure that everything falls within appropriate boundaries
		#and if it doesn't, return -inf which will force a new draw in the MCMC chain (corresponds to a zero probability for the guess)
		if any(t > tmax for t in temps) or any(t < tmin for t in temps) or any(r < 0.05 for r in rad)  or rad[0] > 1.5 or dist < 1/3000 or dist > 1/100:
			return  -np.inf
		if ext == True and (extinct < 0):
			return  -np.inf 
		#and now if we're applying a nonzero non-uniform (gaussian) prior with values we entered 
		elif ext == True:
			pos = SkyCoord(ra*u.deg, dec*u.deg, distance = (1/dist)*u.pc)
			av_guess = bayestar(pos, mode = 'samples') * 3.1 * 0.884
			# print(av_guess)
			av_guess_mu = np.mean(av_guess); av_guess_sig = np.std(av_guess)
			if av_guess_sig == 0:
				av_guess_sig = 0.05
			pp.append(-0.5 * ((extinct - av_guess_mu)/av_guess_sig)**2)

		if prior != 0:
			#unpack the various priors and standard deviations from the input prior variable
			tprior = prior[:nspec]
			tpsig = prior[nspec:2*nspec]
			distprior = [prior[-2]]
			distsig = [prior[-1]]
			eprior = prior[2*nspec]
			epsig = prior[2*nspec+1]
			rprior = prior[2*nspec+2:3*nspec+2]
			rsig = prior[3*nspec+2:4*nspec+2]

			#concatenate the various priors and standard deviations in a different way than they were input (this is so clunky!)
			ps = tprior + [eprior] + rprior + distprior
			ss = tpsig + [epsig] + rsig + distsig

			#now for every prior where the stddev is not zrro calculate the likelihood 
			for k, p in enumerate(ps):
				if p != 0:
					like = -0.5 * ((p0[k] - p)/ss[k])**2
					pp.append(like)

		if rad_prior == True:
			model_radius1 = get_radius(temps[0], matrix)
			model_radius2 = get_radius(temps[1], matrix)

			for k, p in enumerate([model_radius1, model_radius2/model_radius1]):
				like = -0.5 * ((rad[k] - p)/(0.02*p))**2
				# print(model_radius1, model_radius2/model_radius1, rad, like)
				pp.append(like)

		#if all values in the prior array are zero, just return zero (equivalent to just adding one uniformly to the likelihood function)
		return np.sum(pp)

	#now if we're calculating the prior for a system where there is a distance entered but we're not using it (e.g., the current fitting system for stars without a Gaia parallax)
	elif len(p0) == 6 and dist_fit == False:
		#first calculate the radius ratio
		#this is what we actually want to assess, since we technically don't have any constraints on the distance so any absolute radii are essentially nonphysical
		rad = p0[-2]
		rad1 = p0[-3]
		#get the extinction guess if I need it
		if ext == True:
			extinct = p0[-4]
		
		pp = []
		#now assess that all the guess parameters fall within the preset limits and return -inf if they don't
		if any(t > tmax for t in temps) or any(t < tmin for t in temps) or rad < 0.05 or rad1 < 0.05:
			return -np.inf
		if ext == True and extinct < 0:
			return-np.inf
		elif ext == True:
			pos = SkyCoord(ra*u.deg, dec*u.deg, distance = (1/p0[-1])*u.pc)
			av_guess = bayestar(pos, mode = 'samples') * 3.1 * 0.884
			# print(av_guess)
			av_guess_mu = np.mean(av_guess); av_guess_sig = np.std(av_guess)
			if av_guess_sig == 0:
				av_guess_sig = 0.05
			pp.append(-0.5 * ((extinct - av_guess_mu)/av_guess_sig)**2)

		#if there are any non-zero non-uniform priors, we need to assess them 
		#this assumes that all non-uniform priors are gaussian
		if prior != 0:
			#unpack the different values
			tprior = prior[:nspec]
			tpsig = prior[nspec:2*nspec]
			eprior = prior[2*nspec]
			epsig = prior[2*nspec+1]
			rprior = prior[2*nspec+2:3*nspec+1]
			rsig = prior[3*nspec+2:4*nspec+1]

			#re-organize the different values into a values list and a stddev list
			ps = tprior + [eprior] + rprior
			ss = tpsig + [epsig] + rsig

			#evalute the log(prior) value for each defined prior entry
			for k, p in enumerate(ps):
				if p != 0:
					like = -0.5 * ((p0[k] - p)/ss[k])**2
					pp.append(like)

		if rad_prior == True:
			model_radius1 = get_radius(temps[0], matrix)
			model_radius2 = get_radius(temps[1], matrix)
			r = [rad1, rad]
			for k, p in enumerate([model_radius1, model_radius2/model_radius1]):
				like = -0.5 * ((r[k] - p)/(0.02*p))**2
				pp.append(like)
		return np.sum(pp)

	if len(p0) == 8 and dist_fit == True:
		#if i'm fitting for extinction, get Av, radii, and distance from the guess array
		if ext == True:
			extinct = p0[nspec]
			rad = p0[nspec + 1:2*nspec + 1]
			dist = p0[2*nspec + 1]
		#otherwise just get the radii and distance
		else:
			rad = p0[nspec:2*nspec]
			dist = p0[2*nspec + 1]

		rad1 = p0[-3]
		rad2 = p0[-2]
		rad = p0[-4]

		pp = []
		#now check to make sure that everything falls within appropriate boundaries
		#and if it doesn't, return -inf which will force a new draw in the MCMC chain (corresponds to a zero probability for the guess)
		if any(t > tmax for t in temps) or any(t < tmin for t in temps) or any(r < 0.05 for r in [rad, rad1, rad2]) or dist < 1/1000 or dist > 1/100:
			return  -np.inf
		if ext == True and extinct < 0:
			return  -np.inf 

		elif ext == True:
			pos = SkyCoord(ra*u.deg, dec*u.deg, distance = (1/dist)*u.pc)
			av_guess = bayestar(pos, mode = 'samples') * 3.1 * 0.884
			av_guess_mu = np.mean(av_guess); av_guess_sig = np.std(av_guess)
			if av_guess_sig == 0:
				av_guess_sig = 0.05
			pp.append(-0.5 * ((extinct - av_guess_mu)/av_guess_sig)**2)


		#and now if we're applying a nonzero non-uniform (gaussian) prior with values we entered 
		if prior != 0:
			#unpack the various priors and standard deviations from the input prior variable
			tprior = prior[:nspec]
			tpsig = prior[nspec:2*nspec]
			distprior = [prior[-2]]
			distsig = [prior[-1]]
			eprior = prior[2*nspec]
			epsig = prior[2*nspec+1]
			rprior = prior[2*nspec+2:3*nspec+2]
			rsig = prior[3*nspec+2:4*nspec+2]

			#concatenate the various priors and standard deviations in a different way than they were input (this is so clunky!)
			ps = tprior + [eprior] + rprior + distprior
			ss = tpsig + [epsig] + rsig + distsig

			#now for every prior where the stddev is not zrro calculate the likelihood 
			for k, p in enumerate(ps):
				if p != 0:
					like = -0.5 * ((p0[k] - p)/ss[k])**2
					pp.append(like)

		if rad_prior == True:
			model_radius1 = get_radius(temps[0], matrix)
			model_radius2 = get_radius(temps[1], matrix)
			model_radius3 = get_radius(temps[2], matrix)
			r = [rad, rad1, rad2]
			for k, p in enumerate([model_radius1, model_radius2/model_radius1, model_radius3/model_radius1]):
				like = -0.5 * ((r[k] - p)/(0.02*p))**2
				pp.append(like)

			#return the non-zero finite prior
			return  np.sum(pp)


	#now if we're calculating the prior for a system where there is a distance entered but we're not using it (e.g., the current fitting system for stars without a Gaia parallax)
	elif len(p0) == 8 and dist_fit == False:
		#first calculate the radius ratio
		#this is what we actually want to assess, since we technically don't have any constraints on the distance so any absolute radii are essentially nonphysical
		rad1 = p0[-3]
		rad2 = p0[-2]
		rad = p0[-4]
		dist = p0[-1]
		#get the extinction guessif I need it
		if ext == True:
			extinct = p0[-5]

		pp = []

		#now assess that all the guess parameters fall within the preset limits and return -inf if they don't
		if any(t > tmax for t in temps) or any(t < tmin for t in temps) or rad1 < 0.05 or rad2 < 0.05 or rad2 < 0.05 or dist < 0:
			return -np.inf
		if ext == True and extinct < 0:
			return-np.inf
		elif ext == True:
			pos = SkyCoord(ra*u.deg, dec*u.deg, distance = (1/dist)*u.pc)
			av_guess = bayestar(pos, mode = 'samples') * 3.1 * 0.884
			av_guess_mu = np.mean(av_guess); av_guess_sig = np.std(av_guess)
			if av_guess_sig == 0:
				av_guess_sig = 0.05
			pp.append(-0.5 * ((extinct - av_guess_mu)/av_guess_sig)**2)

		#if there are any non-zero non-uniform priors, we need to assess them 
		#this assumes that all non-uniform priors are gaussian
		if prior != 0:
			#unpack the different values
			tprior = prior[:nspec]
			tpsig = prior[nspec:2*nspec]
			eprior = prior[2*nspec]
			epsig = prior[2*nspec+1]
			rprior = prior[2*nspec+2:3*nspec+1]
			rsig = prior[3*nspec+2:4*nspec+1]

			#re-organize the different values into a values list and a stddev list
			ps = tprior + [eprior] + rprior
			ss = tpsig + [epsig] + rsig

			#evalute the log(prior) value for each defined prior entry
			for k, p in enumerate(ps):
				if p != 0:
					like = -0.5 * ((p0[k] - p)/ss[k])**2
					pp.append(like)

		if rad_prior == True:
			model_radius1 = get_radius(temps[0], matrix)
			model_radius2 = get_radius(temps[1], matrix)
			model_radius3 = get_radius(temps[2], matrix)
			r = [rad, rad1, rad2]
			for k, p in enumerate([model_radius1, model_radius2/model_radius1, model_radius3/model_radius1]):
				like = -0.5 * ((r[k] - p)/(0.02*p))**2
				pp.append(like)

		#and return the prior 
		return np.sum(pp)

	else:
		print('P0 doesn\'t match what I was expecting')

def logposterior(p0, fr, nspec, ndust, data, err, broadening, r, specs, ctm, ptm, tmi, tma, vs, tmin, tmax, matrix, ra, dec, wu = 'aa', dust = False, norm = True, prior = 0, a = True, models = 'btsettl', dist_fit = True, rad_prior = False):
	"""The natural logarithm of the joint posterior.
	"""
	lp = logprior(p0, nspec, ndust, tmin, tmax, matrix, ra, dec, prior = prior, ext = a, dist_fit = dist_fit, rad_prior = rad_prior)
	# if the prior is not finite return a probability of zero (log probability of -inf)
	#otherwise call the likelihood function 
	if not np.isfinite(lp):
		return -np.inf
	else:
		lh = loglikelihood(p0, fr, nspec, ndust, data, err, broadening, r, specs, ctm, ptm, tmi, tma, vs, matrix, w = wu, dust = False, norm = True, av = a, optimize = False, models = models)
		# return the likeihood times the prior (log likelihood plus the log prior)
		return lp + lh

def run_emcee(dirname, fname, nwalkers, nsteps, ndim, nburn, pos, fr, nspec, ndust, data, err, broadening, r, specs, value1, ctm, ptm, tmi, tma, vs, title_format, matrix, ra, dec, nthin=10, w = 'aa', du = False, no = True, prior = 0, av = True, models = 'btsettl', dist_fit = True, rad_prior = False):
	"""Run the emcee code to fit a combined data set.
	"""
	#first, define some limits for the prior based on the spectra we've read in
	#initialize a temperature and log(g) array
	t = []
	#go through the input model spectrum dictionary
	for s in specs.keys():
		#for all real spectra (not the wavelength vector entry)
		if not s == 'wl':
			#just take the key and split it to get a teff and a surface gravity for eahc entry
			p = s.split(', ')
			t.append(float(p[0]))

	#just take the extrema of the range of entry values to be the upper and lower limits for the values allowed by the prior
	tmin, tmax = min(t), max(t)

	count = mp.cpu_count()
	# with mp.Pool(processes = count - 2) as pool:
	# 	sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, threads=nwalkers, args=[fr, nspec, ndust, data, err, broadening, r, specs, ctm, ptm, tmi, tma, vs, tmin, tmax, matrix, ra, dec], \
	# 	kwargs={'dust': du, 'norm':no, 'prior':prior, 'a':av, 'models':models, 'dist_fit':dist_fit, 'rad_prior':rad_prior})

	# 	for n, s in enumerate(sampler.sample(pos, iterations = nburn)):
	# 		if n % nthin == 0:
	# 			with open('{}/{}_{}_burnin.txt'.format(dirname,fname, n), 'ab') as f:
	# 				f.write(b"\n")
	# 				np.savetxt(f, s.coords)
	# 				f.close() 

	# 	state = sampler.get_last_sample()
	# 	sampler.reset()

	# 	old_acl = np.inf
	# 	for n, s in enumerate(sampler.sample(state, iterations = nsteps)):
	# 		if n % nthin == 0:
	# 			with open('{}/{}_{}_results.txt'.format(dirname,fname, n), 'ab') as f:
	# 				f.write(b'\n')
	# 				np.savetxt(f, s.coords)
	# 				f.close()
					
	# 			acl = sampler.get_autocorr_time(quiet = True)
	# 			macl = np.mean(acl)

	# 			with open('{}/{}_autocorr.txt'.format(dirname,fname), 'a') as f:
	# 				f.write(str(macl) + '\n')
			
	# 			if not np.isnan(macl):
	# 				converged = np.all(acl * 50 < n)
	# 				converged &= np.all((np.abs(old_acl - acl) / acl) < 0.1)
	# 				if converged == True:
	# 					break

	# 			old_acl = acl
	# 	print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

	# 	samples = sampler.chain[:, :, :].reshape((-1, ndim))

	# 	np.savetxt(os.getcwd() + '/{}/samples.txt'.format(dirname), samples)

	samples = np.genfromtxt(os.getcwd()+'/{}/samples.txt'.format(dirname))

	if ndim == 6:
		samples[:,-1] *= 1e3

		if dist_fit == True:			

			## make the plots not suck

			plt.rcParams['lines.linewidth']   =2
			plt.rcParams['axes.linewidth']    = 1.5
			plt.rcParams['xtick.major.width'] =2
			plt.rcParams['ytick.major.width'] =2
			plt.rcParams['ytick.labelsize'] = 13
			plt.rcParams['xtick.labelsize'] = 13
			plt.rcParams['axes.labelsize'] = 18
			plt.rcParams['legend.numpoints'] = 1
			plt.rcParams['axes.labelweight']='semibold'
			plt.rcParams['mathtext.fontset']='stix'
			plt.rcParams['font.weight'] = 'semibold'
			plt.rcParams['axes.titleweight']='semibold'
			plt.rcParams['axes.titlesize']=9

			figure = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], \
				labels = [r'T$_{eff,1}$', r'T$_{eff,2}$', r'$A_{V}$', r'R$_{1}$', r'R$_{2}$/R$_{1}$', r'$\pi$ (mas)'], show_titles = True,\
				 bins = 50, fill_contours = True, plot_datapoints = False, title_kwargs = {'fontsize':15}, hist_kwargs={"linewidth":2}, smooth = 0.75, title_fmt = title_format)
			
			if not all(v for v in value1) == 0:
				value1[-1] *= 1e3
				# Extract the axes
				axes = np.array(figure.axes).reshape((ndim, ndim))

				# Loop over the diagonal
				for i in range(ndim):
					ax = axes[i, i]
					ax.axvline(value1[i], color="g")

				# Loop over the histograms
				for yi in range(ndim):
					for xi in range(yi):
						ax = axes[yi, xi]
						ax.axvline(value1[xi], color="g")
						ax.axhline(value1[yi], color="g")
						ax.plot(value1[xi], value1[yi], "sg")

			figure.savefig(os.getcwd() + "/{}/plots/{}_corner.pdf".format(dirname, fname))
			plt.close()

		elif dist_fit == False:

			plt.rcParams['lines.linewidth']   =2
			plt.rcParams['axes.linewidth']    = 1.5
			plt.rcParams['xtick.major.width'] =2
			plt.rcParams['ytick.major.width'] =2
			plt.rcParams['ytick.labelsize'] = 13
			plt.rcParams['xtick.labelsize'] = 13
			plt.rcParams['axes.labelsize'] = 18
			plt.rcParams['legend.numpoints'] = 1
			plt.rcParams['axes.labelweight']='semibold'
			plt.rcParams['mathtext.fontset']='stix'
			plt.rcParams['font.weight'] = 'semibold'
			plt.rcParams['axes.titleweight']='semibold'
			plt.rcParams['axes.titlesize']=9

			figure = corner.corner(samples[:,:-1], quantiles=[0.16, 0.5, 0.84], \
				labels = [r'T$_{eff,1}$', r'T$_{eff,2}$', r'A$_{V}$', r'R$_{1}$', r'R$_{2}$/R$_{1}$'], show_titles = True,\
				 bins = 50, fill_contours = True, plot_datapoints = False, title_kwargs = {'fontsize':15}, hist_kwargs={"linewidth":2}, smooth = 0.75, title_fmt = title_format)
			
			if not all(v for v in value1) == 0:
				axes = np.array(figure.axes).reshape((ndim-2, ndim-2))

				# Loop over the diagonal
				for i in range(ndim-1):
					ax = axes[i, i]

					ax.axvline(value1[i], color="g")

				# Loop over the histograms
				for yi in range(ndim-1):
					for xi in range(yi):
						ax = axes[yi, xi]
						ax.axvline(value1[xi], color="g")
						ax.axhline(value1[yi], color="g")
						ax.plot(value1[xi], value1[yi], "sg")

			figure.savefig(os.getcwd() + "/{}/plots/{}_corner.pdf".format(dirname, fname))
			plt.close()

	elif ndim == 8:

			plt.rcParams['lines.linewidth']   =2
			plt.rcParams['axes.linewidth']    = 1.5
			plt.rcParams['xtick.major.width'] =2
			plt.rcParams['ytick.major.width'] =2
			plt.rcParams['ytick.labelsize'] = 13
			plt.rcParams['xtick.labelsize'] = 13
			plt.rcParams['axes.labelsize'] = 18
			plt.rcParams['legend.numpoints'] = 1
			plt.rcParams['axes.labelweight']='semibold'
			plt.rcParams['mathtext.fontset']='stix'
			plt.rcParams['font.weight'] = 'semibold'
			plt.rcParams['axes.titleweight']='semibold'
			plt.rcParams['axes.titlesize']=9

			print(np.shape(samples))
			figure = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], \
				labels = [r'T$_{eff,1}$', r'T$_{eff,2}$', r'T$_{eff,3}$', r'$A_{V}$', r'R$_{1}$', r'R$_{2}$/R$_{1}$', r'R$_{3}$/R$_{1}$', r'$\pi$'], show_titles = True,\
				 bins = 50, fill_contours = True, plot_datapoints = False, title_kwargs = {'fontsize':15}, hist_kwargs={"linewidth":2}, smooth = 0.75, title_fmt = title_format)
			
			if not all(v for v in value1) == 0:
				value1[-1] *= 1e3
				# Extract the axes
				axes = np.array(figure.axes).reshape((ndim-1, ndim-1))

				# Loop over the diagonal
				for i in range(ndim):
					ax = axes[i, i]
					ax.axvline(value1[i], color="g")

				# Loop over the histograms
				for yi in range(ndim):
					for xi in range(yi):
						ax = axes[yi, xi]
						ax.axvline(value1[xi], color="g")
						ax.axhline(value1[yi], color="g")
						ax.plot(value1[xi], value1[yi], "sg")

			figure.savefig(os.getcwd() + "/{}/plots/{}_corner.pdf".format(dirname, fname))
			plt.close()

	else:
		figure = corner.corner(samples, quantiles=[0.16, 0.5, 0.84],labels = [r'T$_{eff,1}$', r'T$_{eff,2}$', r'$A_{V}$' r'$\frac{Rad_{2}}{Rad_{1}}$'], show_titles = True, title_fmt = title_format)

		if not all(v for v in value1) == 0:
			# Extract the axes
			axes = np.array(figure.axes).reshape((ndim-1, ndim-1))

			# Loop over the diagonal
			for i in range(ndim):
				ax = axes[i, i]
				ax.axvline(value1[i], color="g")

			# Loop over the histograms
			for yi in range(ndim):
				for xi in range(yi):
					ax = axes[yi, xi]
					ax.axvline(value1[xi], color="g")
					ax.axhline(value1[yi], color="g")
					ax.plot(value1[xi], value1[yi], "sg")

		figure.savefig(os.getcwd() + "/{}/plots/{}_corner.pdf".format(dirname, fname))
		plt.close()

	return samples

def optimize_fit(dirname, data, err, specs, nwalk, fr, dist_arr, av, res, ctm, ptm, tmi, tma, vs, matrix, ra, dec, cutoff = 2, nspec = 2, nstep = 200, nburn = 20, con = True, models = 'btsettl', err2 = 0, dist_fit = True, rad_prior = False):
	#we're going to do rejection sampling:
	#initialize a set number of random walkers randomly in parameter space (teff 1 + 2, log(g) 1 + 2)
	#for now, assume we've already fit for extinction - should be easy enough to add in an additional parameter eventually
	#do a reduced chi square to inform steps
	#take some assigned number of steps/until we hit convergence (chisq < 2 or something)
	#take the best 50% (or whatever) and use those as the MCMC walkers


	#first, get the temperature and log(g) range from sp by reading in the keys
	#we need this so we can initialize the random walkers in the correct parameter space
	t = []
	for s in specs.keys():
		if not s == 'wl':
			a = s.split(', ')
			t.append(float(a[0]))

	tmin, tmax = min(t), max(t)

	rmin = 0.05; rmax = 1 #min and max radii in solar radius

	# #now we have the limits, we need to initialize the random walkers over the parameter space
	# #we need to assign four numbers: the two temps and the two log(g)s
	# #so randomly distribute the assigned number of walkers over the parameter space
	# #making sure that the secondary temperature is always less than the primary

	t1 = np.random.uniform(tmin, tmax, nwalk)
	t2 = []
	for tt in t1:
		tt2 = np.random.uniform(tmin, tt)
		t2.append(tt2)

	if nspec == 3:
		t3 = []
		for tt in t2:
			tt3 = np.random.uniform(tmin, tt)
			t3.append(tt3)

	e1 = np.random.uniform(0.1, 0.5, nwalk)

	if not type(dist_arr) == bool:
		rg1 = np.random.uniform(rmin, rmax, nwalk)
		rg2 = []
		for r in rg1:
			r2 = np.random.uniform(rmin, r)
			rg2.append(r2/r)
		if nspec == 3:
			rg3 = []
			for r in rg2:
				r3 = np.random.uniform(rmin, r)
				rg3.append(r3/r)

	else:
		rad = np.random.uniform(rmin, rmax)

	dist = np.random.normal(dist_arr[0], dist_arr[1], nwalk)

	dist = np.abs(dist)
	with mp.Pool(processes = 15) as pool:

		if nspec == 2:
			out = [pool.apply_async(fit_spec, \
					args = (n, dirname, data[0], data[1], err, [min(data[0]), max(data[0])], [t1[n], t2[n]], [e1[n], av[0], av[1]], [rg1[n], rg2[n]], fr, specs, [tmin, tmax], [dist[n], dist_arr[0], dist_arr[1]], ctm, ptm, tmi, tma, vs, matrix, ra, dec), \
					kwds = dict(steps = nstep, models = models, dist_fit = dist_fit, rad_prior = rad_prior)) for n in range(nwalk)]
		elif nspec == 3:
			out = [pool.apply_async(fit_spec, \
				args = (n, dirname, data[0], data[1], err, [min(data[0]), max(data[0])], [t1[n], t2[n], t3[n]], [e1[n], av[0], av[1]], [rg1[n], rg2[n], rg3[n]], fr, specs, [tmin, tmax], [dist[n], dist_arr[0], dist_arr[1]], ctm, ptm, tmi, tma, vs, matrix, ra, dec), \
				kwds = dict(nspec = 3, steps = nstep, models = models, dist_fit = dist_fit, rad_prior = rad_prior)) for n in range(nwalk)]

		a = [o.get() for o in out]

		for line in a:
			gi = line[0]; cs = line[1]

			with open(dirname + '/optimize_res.txt', 'a') as f:
				f.write(gi)
			with open(dirname + '/optimize_cs.txt', 'a') as f:
				f.write(str(cs) + '\n')

	return 

def plot_fit(run, data, sp, fr, ctm, ptm, tmi, tma, vs, matrix, models = 'btsettl', dist_fit = True):
	'''
	Plots for after the optimization step
	'''
	#find all the chisq and parameter files 
	cs_files = glob(run + '/chisq*txt')
	walk_files = glob(run + '/params*txt')

	#initialize a bunch of figures 
	#there was deifnitely an easier way to do this, but whatever
	fig1, ax1 = plt.subplots()
	fig2, ax2 = plt.subplots()
	fig3, ax3 = plt.subplots()
	fig4, ax4 = plt.subplots()
	fig5, ax5 = plt.subplots()
	fig6, ax6 = plt.subplots()

	#for each set of parameter files
	for f in walk_files:
		#open the file
		results = np.genfromtxt(f, dtype = 'S')

		#initialize a bunch of arrays to hold the parameters
		tem1, tem2, ext, rad1, rad2, dist = [], [], [], [], [], []

		#read in the file after saving each line
		for n, line in enumerate(results):
			try:
				tem1.append(float(line[0])); tem2.append(float(line[1]))
				ext.append(float(line[2])); rad1.append(float(line[3]))
				rad2.append(float(line[4])), dist.append(float(line[5]))

				#if we're on the last line, read in the final values to separate variables as well, since these are the best-fit values
				if n == len(results)-1:
					tt1, tt2, te, tr1, tr2, d = float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])
			#if there's only a radius ratio and no distance, the try clause above won't work so read in the shorter line correctly here
			except:
				tem1.append(float(line[0])); tem2.append(float(line[1]))
				ext.append(float(line[2])); rad1.append(float(line[3]))

		#plot the steps as a function of length for each variable
		ax1.plot(range(len(tem1)), tem1, color = 'k', alpha = 0.5)
		ax2.plot(range(len(tem2)), tem2, color = 'k', alpha = 0.5)
		ax3.plot(range(len(ext)), ext, color = 'k', alpha = 0.5)
		ax4.plot(range(len(rad1)), rad1, color = 'k', alpha = 0.5)
		try:
			ax5.plot(range(len(rad2)), rad2, color = 'k', alpha = 0.5)
			ax6.plot(range(len(dist)), dist, color = 'k', alpha = 0.5)
		except:
			pass

	#once all the chains have been plotted, turn on minorticks first 
	plt.minorticks_on()
	#now try to make all the figures
	try:
		figs = [fig1, fig2, fig3, fig4, fig5, fig6]

		#format everything nicely for each figure and save it
		for n, a in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
			labels = ['teff1', 'teff2', 'Av', 'rad1', 'rad2', 'dist']
			a.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
			a.tick_params(bottom=True, top =True, left=True, right=True)
			a.tick_params(which='both', labelsize = "large", direction='in')
			a.tick_params('both', length=8, width=1.5, which='major')
			a.tick_params('both', length=4, width=1, which='minor')
			a.set_xlabel('Step number', fontsize = 13)
			a.set_ylabel('{}'.format(labels[n]), fontsize = 13)
			plt.tight_layout()
			figs[n].savefig(run + '/plots/fit_res_{}.png'.format(labels[n]))
			plt.close()
	#if there's no distance some of those arrays won't be defined, so come down here and make fewer figures		
	except:
		figs = [fig1, fig2, fig3, fig4, fig5]

		#format everything nicely for each figure and save it
		for n, a in enumerate([ax1, ax2, ax3, ax4, ax5]):
			labels = ['teff1', 'teff2', 'Av', 'rad1', 'rad2']
			a.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
			a.tick_params(bottom=True, top =True, left=True, right=True)
			a.tick_params(which='both', labelsize = "large", direction='in')
			a.tick_params('both', length=8, width=1.5, which='major')
			a.tick_params('both', length=4, width=1, which='minor')
			a.set_xlabel('Step number', fontsize = 13)
			a.set_ylabel('{}'.format(labels[n]), fontsize = 13)
			plt.tight_layout()
			figs[n].savefig(run + '/plots/fit_res_{}.png'.format(labels[n]))
			plt.close()


	#read in the best-fit parameters from each run 
	chisqs, pars = np.genfromtxt(run + '/optimize_cs.txt'), np.genfromtxt(run + '/optimize_res.txt')

	#make a best-fit data set using the parameters from the chi^2 minimum 
	if dist_fit == True and len(pars[0,:]) == 6:
		tt1, tt2, te, rad1, rad2, dist = pars[np.where(chisqs == min(chisqs))][0]
		tl1, tl2 = [get_logg(t, matrix) for t in [tt1, tt2]]
		w, spe, p1, p2, p3 = make_composite([tt1, tt2], [tl1, tl2], [rad1, rad2], dist, fr[2], fr[5], [min(data[0]), max(data[0])], sp, ctm, ptm, tmi, tma, vs, models = models)
	elif len(pars[0,:]) == 6 and dist_fit == False:
		tt1, tt2, te, rad1, rad2, dist = pars[np.where(chisqs == min(chisqs))][0]
		tl1, tl2 = [get_logg(t, matrix) for t in [tt1, tt2]]
		ratio = rad2/rad1
		w, spe, p1, p2, p3 = make_composite([tt1, tt2], [tl1, tl2], [ratio], False, fr[2], fr[5], [min(data[0]), max(data[0])], sp, ctm, ptm, tmi, tma, vs, models = models)
	else:
		tt1, tt2, te, ratio1 = pars[np.where(chisqs == min(chisqs))][0]
		tl1, tl2 = [get_logg(t, matrix) for t in [tt1, tt2]]
		w, spe, p1, p2, p3 = make_composite([tt1, tt2], [tl1, tl2], [ratio1], False, fr[2], fr[5], [min(data[0]), max(data[0])], sp, ctm, ptm, tmi, tma, vs, models = models)

	#extinct the best-fit spectrum
	spe = extinct(w, spe, te)

	#interpoalte the best-fit spectrum onto the data wavelength
	itep = interp1d(w, spe)
	spe = itep(data[0]*1e4)

	#normalize the best-fit spectrum to the data
	spe *= np.median(data[1])/np.median(spe)

	#plot the data and the best-fit spectrum, and save the figure
	plt.figure()
	plt.minorticks_on()
	plt.plot(data[0]*1e4, data[1], color = 'navy', linewidth = 1)
	plt.plot(data[0]*1e4, spe, color = 'xkcd:sky blue', label = 'model: {:.0f} + {:.0f}; {:.1f} + {:.1f}; {:.2f}'.format(tt1, tt2, tl1, tl2, te), linewidth = 1)
	plt.xlim(max(min(w), min(data[0]*1e4)), min(max(w), max(data[0])*1e4))
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.xlabel('Wavelength (A)', fontsize = 13)
	plt.ylabel('Normalized flux', fontsize = 13)
	plt.legend(loc = 'best', fontsize = 13)
	plt.tight_layout()
	plt.savefig(run + '/plots/bestfit_spec.pdf')
	plt.close()

	return

def plot_fit3(run, data, sp, fr, ctm, ptm, tmi, tma, vs, matrix, models = 'btsettl', dist_fit = True):
	'''
	Plots for after the optimization step
	'''
	#find all the chisq and parameter files 
	cs_files = glob(run + '/chisq*txt')
	walk_files = glob(run + '/params*txt')

	#initialize a bunch of figures 
	#there was deifnitely an easier way to do this, but whatever
	fig1, ax1 = plt.subplots()
	fig2, ax2 = plt.subplots()
	fig3, ax3 = plt.subplots()
	fig4, ax4 = plt.subplots()
	fig5, ax5 = plt.subplots()
	fig6, ax6 = plt.subplots()
	fig7, ax7 = plt.subplots()
	fig8, ax8 = plt.subplots()

	#for each set of parameter files
	for f in walk_files:
		#open the file
		results = np.genfromtxt(f, dtype = 'S')

		#initialize a bunch of arrays to hold the parameters
		tem1, tem2, tem3, ext, rad1, rad2, rad3, dist = [], [], [], [], [], [], [], []

		#read in the file after saving each line
		for n, line in enumerate(results):
			try:
				tem1.append(float(line[0])); tem2.append(float(line[1])); tem3.append(float(line[2]))
				ext.append(float(line[3])); rad1.append(float(line[4]))
				rad2.append(float(line[5])); rad3.append(float(line[6])); dist.append(float(line[7]))

				#if we're on the last line, read in the final values to separate variables as well, since these are the best-fit values
				if n == len(results)-1:
					tt1, tt2, tt3, te, tr1, tr2, tr3, d = float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7])
			#if there's only a radius ratio and no distance, the try clause above won't work so read in the shorter line correctly here
			except:
				print('error in reading in chisq and parameter files')

		#plot the steps as a function of length for each variable
		ax1.plot(range(len(tem1)), tem1, color = 'k', alpha = 0.5)
		ax2.plot(range(len(tem2)), tem2, color = 'k', alpha = 0.5)
		ax3.plot(range(len(tem3)), tem3, color = 'k', alpha = 0.5)
		ax4.plot(range(len(ext)), ext, color = 'k', alpha = 0.5)
		ax5.plot(range(len(rad1)), rad1, color = 'k', alpha = 0.5)
		ax6.plot(range(len(rad2)), rad2, color = 'k', alpha = 0.5)
		ax7.plot(range(len(rad3)), rad3, color = 'k', alpha = 0.5)
		ax8.plot(range(len(dist)), dist, color = 'k', alpha = 0.5)

	#once all the chains have been plotted, turn on minorticks first 
	plt.minorticks_on()
	#now try to make all the figures
	figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]

	#format everything nicely for each figure and save it
	for n, a in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]):
		labels = ['teff1', 'teff2', 'teff3', 'Av', 'rad1', 'rad2', 'rad3', 'dist']
		a.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		a.tick_params(bottom=True, top =True, left=True, right=True)
		a.tick_params(which='both', labelsize = "large", direction='in')
		a.tick_params('both', length=8, width=1.5, which='major')
		a.tick_params('both', length=4, width=1, which='minor')
		a.set_xlabel('Step number', fontsize = 13)
		a.set_ylabel('{}'.format(labels[n]), fontsize = 13)
		plt.tight_layout()
		figs[n].savefig(run + '/plots/fit_res_{}.png'.format(labels[n]))
		plt.close()


	#read in the best-fit parameters from each run 
	chisqs, pars = np.genfromtxt(run + '/optimize_cs.txt'), np.genfromtxt(run + '/optimize_res.txt')

	#make a best-fit data set using the parameters from the chi^2 minimum 
	tt1, tt2, tt3, te, rad1, rad2, rad3, dist = pars[np.where(chisqs == min(chisqs))][0]
	best_lg = [get_logg(t, matrix) for t in [tt1, tt2, tt3]]
	if dist_fit == True and len(pars[0,:]) == 8:
		w, spe, p1, p2, p3 = make_composite([tt1, tt2, tt3], best_lg, [rad1, rad2, rad3], dist, fr[2], fr[5], [min(data[0]), max(data[0])], sp, ctm, ptm, tmi, tma, vs, models = models)
	elif len(pars[0,:]) == 8 and dist_fit == False:
		w, spe, p1, p2, p3 = make_composite([tt1, tt2, tt3], best_lg, [rad2, rad3], False, fr[2], fr[5], [min(data[0]), max(data[0])], sp, ctm, ptm, tmi, tma, vs, models = models)
	else:
		print('error in producing spectrum!')
	#extinct the best-fit spectrum
	spe = extinct(w, spe, te)

	#interpoalte the best-fit spectrum onto the data wavelength
	itep = interp1d(w, spe)
	spe = itep(data[0]*1e4)

	#normalize the best-fit spectrum to the data
	spe *= np.median(data[1])/np.median(spe)

	#plot the data and the best-fit spectrum, and save the figure
	plt.figure()
	plt.minorticks_on()
	plt.plot(data[0]*1e4, data[1], color = 'navy', linewidth = 1)
	plt.plot(data[0]*1e4, spe, color = 'xkcd:sky blue', label = 'model: {:.0f} + {:.0f} + {:.0f}'.format(tt1, tt2, tt3), linewidth = 1)
	plt.xlim(max(min(w), min(data[0]*1e4)), min(max(w), max(data[0])*1e4))
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.xlabel('Wavelength (A)', fontsize = 13)
	plt.ylabel('Normalized flux', fontsize = 13)
	plt.legend(loc = 'best', fontsize = 13)
	plt.tight_layout()
	plt.savefig(run + '/plots/bestfit_spec.pdf')
	plt.close()

	return

def plot_results(fname, sample, run, data, sp, fr, ctm, ptm, tmi, tma, vs, real_val, distance, matrix, models = 'btsettl', res = 1000, dist_fit = True, tell = False):
	#unpack the sample array
	t1, t2, av, r1, r2, pl = sample.T
	l1 = [get_logg(t, matrix) for t in t1]; l2 = [get_logg(t, matrix) for t in t2]
	#calculate the radius ratio
	ratio = r2/r1

	#calculate the median value for all the emcee chains
	a = np.median(sample, axis = 0)
	#draw a random sample of parameters from the emcee draws 
	random_sample = sample[np.random.choice(len(sample), size = 100, replace = False), :]

	############
	#FIND AND PLOT THE BEST-FIT PARAMETERS IN THE BIMODAL DISTRIBUTIONS
	###########
	#define the number of bins to use to create the PDFs
	nbins = 75

	#initialize some arrays 
	#*bins defines the histogram bins; *count is the PDF for the bins (non-normalized)
	t1_bins, t2_bins = np.linspace(min(t1), max(t1), nbins), np.linspace(min(t2), max(t2), nbins)
	r1_bins, r2_bins = np.linspace(min(r1), max(r1), nbins), np.linspace(min(r2), max(r2), nbins)
	ratio_bins = np.linspace(min(ratio), max(ratio), nbins); ratio_count = np.zeros(len(ratio_bins))

	t1_count, t2_count = np.zeros(len(t1_bins)), np.zeros(len(t2_bins))
	r1_count, r2_count = np.zeros(len(r1_bins)), np.zeros(len(r2_bins))

	#fill up the count bins 
	#temperature 1
	for t in t1:
		for b in range(len(t1_bins) - 1):
			if t1_bins[b] <= t < t1_bins[b + 1]:
				t1_count[b] += 1

	#Teff 2
	for t in t2:
		for b in range(len(t2_bins) - 1):
			if t2_bins[b] <= t < t2_bins[b + 1]:
				t2_count[b] += 1

	#radius 1
	for r in r1:
		for b in range(len(r1_bins) - 1):
			if r1_bins[b] <= r < r1_bins[b + 1]:
				r1_count[b] += 1

	#Radius 2
	for r in r2:
		for b in range(len(r2_bins) - 1):
			if r2_bins[b] <= r < r2_bins[b + 1]:
				r2_count[b] += 1

	#radius ratio
	for r in ratio:
		for b in range(len(ratio_bins) - 1):
			if ratio_bins[b] <= r < ratio_bins[b + 1]:
				ratio_count[b] += 1

	#####
	#Temperature 1
	#####
	try:
		#find the local min between the two gaussian temperature measurements 
		t1_localmin = int(np.mean([np.where(t1_count[np.where((t1_bins > min(t1)) & (t1_bins < max(t1)))] < 0.5*max(t1_count))]))
		#fit a bimodal gaussian to the distribution
		fit_t1, cov = curve_fit(bimodal, t1_bins, t1_count, [np.mean(t1_bins[t1_localmin:]), np.std(t1_bins[t1_localmin:]), max(t1_count[t1_localmin:]),\
			 np.mean(t1_bins[:t1_localmin]), np.std(t1_bins[:t1_localmin]), max(t1_count[:t1_localmin])])

		#plot the bimodal distribution, the best fit, and the corresponding component Gaussians
		plt.figure()
		plt.hist(t1, bins = t1_bins)
		plt.axvline(t1_bins[t1_localmin], color = 'k', linewidth = 2)
		plt.plot(t1_bins, t1_count)
		plt.plot(t1_bins, bimodal(t1_bins, *fit_t1), color = 'b')
		plt.plot(t1_bins, gauss(t1_bins, *fit_t1[:3]))
		plt.plot(t1_bins, gauss(t1_bins, *fit_t1[3:]))
		plt.savefig(run + '/plots/bimodal_test_T1.pdf')

		#calculate the area under each curve as a percentage
		t1_v1 = np.trapz(gauss(t1_bins, *fit_t1[:3]))/np.trapz(bimodal(t1_bins, *fit_t1))
		t1_v2 = np.trapz(gauss(t1_bins, *fit_t1[3:]))/np.trapz(bimodal(t1_bins, *fit_t1))

		#pick the gaussian with the largest area and set the appropriate parameters to the corresponding values as established from the fit
		if t1_v1 > t1_v2:
			a[0] = fit_t1[0]; sigma_t1 = fit_t1[1]; 
		else:
			a[0] = fit_t1[3]; sigma_t1 = fit_t1[4]
		p_t1 = max(t1_v1, t1_v2)
	#if none of that works because it's not bimodal, just set the stddev to 0 and the best-fit value will stay at the median value from emcee	
	except:
		sigma_t1 = 0
		pass;

	#####
	#Temperature 2
	#####
	try:
		t2_localmin = int(np.mean([np.where(t2_count[np.where((t2_bins > min(t2)) & (t2_bins < max(t2)))] < 0.5*max(t2_count))]))

		fit_t2, cov = curve_fit(bimodal,t2_bins, t2_count, [np.mean(t2_bins[t2_localmin:]), np.std(t2_bins[t2_localmin:]), max(t2_count[t2_localmin:]), \
				np.mean(t2_bins[:t2_localmin]), np.std(t2_bins[:t2_localmin]), max(t2_count[:t2_localmin])])

		plt.figure()
		plt.hist(t2, bins = t2_bins)
		plt.axvline(t2_bins[t2_localmin], color = 'k', linewidth = 2)
		plt.plot(t2_bins, t2_count)
		plt.plot(t2_bins, bimodal(t2_bins, *fit_t2), color = 'b')
		plt.plot(t2_bins, gauss(t2_bins, *fit_t2[:3]))
		plt.plot(t2_bins, gauss(t2_bins, *fit_t2[3:]))
		plt.savefig(run + '/plots/bimodal_test_T2.pdf')


		t2_v1 = np.trapz(gauss(t2_bins, *fit_t2[:3]))/np.trapz(bimodal(t2_bins, *fit_t2))
		t2_v2 = np.trapz(gauss(t2_bins, *fit_t2[3:]))/np.trapz(bimodal(t2_bins, *fit_t2))
		if t2_v1 > t2_v2:
			a[1] = fit_t2[0]; sigma_t2 = fit_t2[1]; 
		else:
			a[1] = fit_t2[3]; sigma_t2 = fit_t2[4]
		p_t2 = max(t2_v1, t2_v2)
	except:
		sigma_t2 = 0
		pass;

	#####
	#Radius 1
	#####
	try:
		r1_localmin = int(np.mean([np.where(r1_count[np.where((r1_bins > min(r1)) & (r1_bins < max(r1)))] < 0.5*max(r1_count))]))
		fit_r1, cov = curve_fit(bimodal, r1_bins, r1_count, [np.mean(r1_bins[r1_localmin:]), np.std(r1_bins[r1_localmin:]), max(r1_count[r1_localmin:]),\
			 np.mean(r1_bins[:r1_localmin]), np.std(r1_bins[:r1_localmin]), max(r1_count[:r1_localmin])])

		plt.figure()
		plt.hist(r1, bins = r1_bins)
		plt.axvline(r1_bins[r1_localmin], color = 'k', linewidth = 2)
		plt.plot(r1_bins, r1_count)
		plt.plot(r1_bins, bimodal(r1_bins, *fit_r1), color = 'b')
		plt.plot(r1_bins, gauss(r1_bins, *fit_r1[:3]))
		plt.plot(r1_bins, gauss(r1_bins, *fit_r1[3:]))
		plt.savefig(run + '/plots/bimodal_test_R1.pdf')

		r1_v1 = np.trapz(gauss(r1_bins, *fit_r1[:3]))/np.trapz(bimodal(r1_bins, *fit_r1))
		r1_v2 = np.trapz(gauss(r1_bins, *fit_r1[3:]))/np.trapz(bimodal(r1_bins, *fit_r1))
		if r1_v1 > r1_v2:
			a[5] = fit_r1[0]; sigma_r1 = fit_r1[1]; 
		else:
			a[5] = fit_r1[3]; sigma_r1 = fit_r1[4]
		p_r1 = max(r1_v1, r1_v2)
	except:
		sigma_r1 = 0
		pass;

	#####
	#Radius ratio
	#####
	try:
		r2_localmin = int(np.mean([np.where(r2_count[np.where((r2_bins > min(r2)) & (r2_bins < max(r2)))] < 0.5*max(r2_count))]))
		fit_r2, cov = curve_fit(bimodal,r2_bins, r2_count, [np.mean(r2_bins[r2_localmin:]), np.std(r2_bins[r2_localmin:]), max(r2_count[r2_localmin:]), \
				np.mean(r2_bins[:r2_localmin]), np.std(r2_bins[:r2_localmin]), max(r2_count[:r2_localmin])])

		plt.figure()
		plt.hist(r2, bins = r2_bins)
		plt.axvline(r2_bins[r2_localmin], color = 'k', linewidth = 2)
		plt.plot(r2_bins, r2_count)
		plt.plot(r2_bins, bimodal(r2_bins, *fit_r2), color = 'b')
		plt.plot(r2_bins, gauss(r2_bins, *fit_r2[:3]))
		plt.plot(r2_bins, gauss(r2_bins, *fit_r2[3:]))
		plt.savefig(run + '/plots/bimodal_test_R2R1.pdf')

		r2_v1 = np.trapz(gauss(r2_bins, *fit_r2[:3]))/np.trapz(bimodal(r2_bins, *fit_r2))
		r2_v2 = np.trapz(gauss(r2_bins, *fit_r2[3:]))/np.trapz(bimodal(r2_bins, *fit_r2))
		if r2_v1 > r2_v2:
			a[6] = fit_r2[0]; sigma_r2 = fit_r2[1]; 
		else:
			a[6] = fit_r2[3]; sigma_r2 = fit_r2[4]
		p_r2 = max(r2_v1, r2_v2)
	except:
		sigma_r2 = 0
		pass;


	############
	#CREATE THE PHOTOMETRY AND CONTRAST PLOTS
	###########

	# a = np.median(sample, axis = 0)


	plt.minorticks_on()
	wl, spec, err = data
	err /= np.median(spec); spec /= np.median(spec)

	#unpack the best-fit stellar parameters 
	if len(a) == 6:
		tt1, tt2, te, rad1, rad2, plx = np.median(sample, axis = 0)
		logg_guess = [get_logg(t, matrix) for t in [tt1, tt2]]
		ratio1 = rad2
	else:
		tt1, tt2, tl1, tl2, te, ratio1 = a

	if len(a) == 6:
		ww, ss, c, pcwl, phot = make_composite([tt1, tt2], logg_guess, [rad1, rad2], plx, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = False)
		
		w, spe, pri_spec, sec_spec, pri_mag, sec_mag, gpm, gsm, gm = make_composite([tt1, tt2], logg_guess, [rad1, rad2], plx, fr[2], fr[5], [3000, 24000], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		new_wl, new_prispec = redres(w, pri_spec, 250)
		new_wl, new_sp = redres(w, spe, 250)
		new_wl, new_secspec = redres(w, sec_spec, 250)

		new_wl, new_sp, new_prispec, new_secspec = new_wl[np.where((new_wl >= 5315) & (new_wl <= 23652))], new_sp[np.where((new_wl >= 5315) & (new_wl <= 23652))], new_prispec[np.where((new_wl >= 5315) & (new_wl <= 23652))], new_secspec[np.where((new_wl >= 5315) & (new_wl <= 23652))]

		zp = [2.854074834606756e-09,1.940259205607388e-09,1.359859453789013e-09,3.1121838042516567e-10,1.1353317746392182e-10, 4.279017715611946e-11]
		phot_cwl = [6175., 7489., 8946., 12350., 16620., 21590.]
		phot_width = np.array([[6175-5415, 6989-6175], [7489-6689, 8389-7489], [8946-7960, 10833-8946], [12350-10806, 14067-12350], [16620-14787, 18231-16620], [21590-19543, 23552-21590]]).T

		phot = -2.5*np.log10(extinct(np.array(phot_cwl[:len(phot)]), 10**(-0.4*np.array(phot)), te))
		new_spe = extinct(np.array(new_wl), new_sp, te)

		fig,ax = plt.subplots(nrows = 3, gridspec_kw = dict(hspace = 0, height_ratios = [3, 1.75, 1]), sharex = True, figsize = (7 , 6))
		e = ax[0].scatter(phot_cwl[:len(phot)], 10**(-0.4*phot)*zp[:len(phot)], color = 'seagreen', s = 100, marker = '.', label = 'Composite phot.')
		ax[0].errorbar(phot_cwl[:len(phot)], 10**(-0.4*phot)*zp[:len(phot)], xerr = phot_width[:len(phot)], color= 'seagreen', zorder = 0, linestyle = 'None')
		b = ax[0].scatter(phot_cwl[:len(phot)], 10**(-0.4*fr[3])*zp[:len(phot)], linestyle = 'None', color = 'k', marker = '.', s = 100, label = 'Data phot.')
		m = ax[0].plot(new_wl, new_spe, color = 'seagreen', linewidth = 1, zorder = 0, alpha = 0.5)
		plt.minorticks_on()
		ax[0].set_xscale('log')
		ax[0].set_yscale('log')
		ax[0].tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		ax[0].tick_params(bottom=True, top =True, left=True, right=True)
		ax[0].tick_params(which='both', labelsize = 12, direction='in')
		ax[0].tick_params('both', length=8, width=1.5, which='major')
		ax[0].tick_params('both', length=4, width=1, which='minor')
		ax[0].set_ylabel('{}'.format(r'Flux (erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)'), fontsize = 12)

		f = ax[1].scatter(ctm[3], c, color = 'blue', marker = 'v', label = 'Model contrast', zorder = 2)
		ax[1].errorbar(ctm[3], fr[0], yerr = fr[1], linestyle = 'None', capsize = 4, capthick = 2, color = 'k', marker = 'v', zorder = 1)
		g = ax[1].scatter(ctm[3], fr[0], color = 'k', marker = 'v', label = 'Data contrast', zorder = 1)
		ax[1].plot(new_wl, 2.5*np.log10((new_prispec)) - 2.5*np.log10((new_secspec)), color = 'blue', linewidth = 1, zorder = 0, alpha = 0.5)
		ax[1].set_ylabel(r'$\Delta$ mag', fontsize = 12)

		plt.minorticks_on()
		ax[1].tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		ax[1].tick_params(bottom=True, top =True, left=True, right=True)
		ax[1].tick_params(which='both', labelsize = 12, direction='in')
		ax[1].tick_params('both', length=8, width=1.5, which='major')
		ax[1].tick_params('both', length=4, width=1, which='minor')

		ax[2].scatter(phot_cwl[:len(phot)], phot[:len(phot)]-fr[3][:len(phot)], color = 'seagreen', marker = 'x', s = 50, label = 'Phot. resid.')
		ax[2].axhline(0, color = '0.3', linestyle = '--', linewidth = 2, label = 'No resid.')
		ax[2].scatter(ctm[3], np.array(fr[0])-np.array(c), color = 'blue', marker = 'x', label = 'Cont. resid.',s = 50)
		plt.minorticks_on()
		ax[2].tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		ax[2].tick_params(bottom=True, top =True, left=True, right=True)
		ax[2].tick_params(which='both', labelsize = 12, direction='in')
		ax[2].tick_params('both', length=8, width=1.5, which='major')
		ax[2].tick_params('both', length=4, width=1, which='minor')
		ax[2].set_xlabel(r'Wavelength (\AA)', fontsize = 12)
		ax[2].set_ylabel('{}'.format(r'Resid. (mag)'), fontsize = 12)

		fig.align_ylabels(ax)

		handles, labels = plt.gca().get_legend_handles_labels()
		handles.extend([e,b,f,g])

		ax[0].legend(handles = handles, loc = 'best', fontsize = 10, ncol = 2)

		plt.tight_layout()
		plt.savefig(run + '/plots/{}_phot_scatter.pdf'.format(fname))
		plt.close()

	else:
		ww, ss, c, pcwl, phot = make_composite([tt1, tt2], logg_guess, [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = False)
		fig,ax = plt.subplots(nrows = 2, gridspec_kw = dict(hspace = 0, height_ratios = [3, 1]), sharex = True, figsize = (7,6))
		f = ax[0].scatter(ctm[3], c, color = 'blue', marker = 'v', label = 'Model contrast', zorder = 2)
		g = ax[0].errorbar(ctm[3], fr[0], yerr = fr[1], linestyle = 'None', capsize = 4, capthick = 2, color = 'k', marker = 'v', label = 'Data contrast', zorder = 1)
		ax[0].set_ylabel('Contrast (mag)', fontsize = 12)
		plt.minorticks_on()
		ax[0].tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		ax[0].tick_params(bottom=True, top =True, left=True, right=True)
		ax[0].tick_params(which='both', labelsize = 14, direction='in')
		ax[0].tick_params('both', length=8, width=1.5, which='major')
		ax[0].tick_params('both', length=4, width=1, which='minor')

		d = ax[1].axhline(0, color = '0.3', linestyle = '--', linewidth = 2, label = 'No resid.')
		h = ax[1].scatter(ctm[3], np.array(fr[0])-np.array(c), color = 'blue', marker = 'x', label = 'Cont. resid.',s = 50)
		plt.minorticks_on()
		ax[1].tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		ax[1].tick_params(bottom=True, top =True, left=True, right=True)
		ax[1].tick_params(which='both', labelsize = 14, direction='in')
		ax[1].tick_params('both', length=8, width=1.5, which='major')
		ax[1].tick_params('both', length=4, width=1, which='minor')
		ax[1].set_xlabel(r'Wavelength ($\AA$)', fontsize = 12)
		ax[1].set_ylabel('{}'.format(r'Residual (mag)'), fontsize = 12)
		ax[1].set_xscale('log')
		fig.align_ylabels(ax)

		handles, labels = plt.gca().get_legend_handles_labels()
		handles.extend([f,g])

		ax[0].legend(handles = handles, loc = 'best', fontsize = 10, ncol = 2)

		plt.tight_layout()
		plt.savefig(run + '/plots/{}_phot_scatter.pdf'.format(fname))
		plt.close()

	############
	#CREATE THE COMPOSITE + COMPONENET + DATA PLOT (all_spec)
	###########

	#contrast and phot (unresolved photometry) in flux
	#pwl is the central wavelength for the photometric filters
	if len(a) == 6 and dist_fit == True:
		w, spe, pri_spec, sec_spec, pri_mag, sec_mag, gpm, gsm, gm = make_composite([tt1, tt2], logg_guess, [rad1, rad2], plx, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
	else:	
		w, spe, pri_spec, sec_spec, pri_mag, sec_mag, gpm, gsm, gm = make_composite([tt1, tt2], logg_guess, [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)

	spe = extinct(w, spe, te)
	pri_spec = extinct(w, pri_spec, te)
	sec_spec = extinct(w, sec_spec, te)

	itep = interp1d(w, spe)
	spe = itep(wl*1e4)

	if len(a) == 6:
		ratio1 = rad2

	pri_ratio =  (np.median(spec)/np.median(spe))
	sec_ratio = (np.median(spec)/np.median(spe))

	i1 = interp1d(w, pri_spec)
	pri_spec = i1(wl*1e4)
	pri_spec *= pri_ratio
	i2 = interp1d(w, sec_spec)
	sec_spec = i2(wl*1e4)
	sec_spec *= sec_ratio

	spe *= np.median(spec)/np.median(spe)
	spec = norm_spec(wl, spe, spec) 

	if tell == True:
		regions = [[6860, 6880], [7600, 7660], [8210, 8240]]

	with open(run + '/params.txt', 'w') as f:
		if len(a) == 6 and dist_fit == True:
			f.write('teff: {} +/- {} + {} +/- {}\nradius: {} +/- {} + {} +/- {}\nextinction: {}\nparallax: {}\nprimary Kep mag:{}\nsecondary Kep mag:{}'.format(tt1, sigma_t1, tt2, sigma_t2, rad1, sigma_r1, rad2, sigma_r2, te, plx, pri_mag, sec_mag))
		else:
			f.write('teff: {} +/- {} + {} +/- {}\nradius: {} +/- {}\nextinction: {}\nprimary Kep mag:{}\nsecondary Kep mag:{}'.format(tt1, sigma_t1, tt2, sigma_t2, rad2, sigma_r2, te, pri_mag, sec_mag))
	#make a nice figure with the different spectra
	fig, [ax, ax1] = plt.subplots(nrows = 2, gridspec_kw = dict(hspace = 0, height_ratios = [3, 1]), sharex = True, figsize = (7 , 6))

	if any(v != 0 for v in real_val):
		ax.plot(wl*1e4, spec, linewidth = 1, label = 'Data: {:.0f}+{:.0f}K'.format(real_val[0], real_val[1]), color = 'k', zorder = 4)
	else:
		ax.plot(wl*1e4, spec, linewidth = 1, label = 'Data', color = 'k', zorder = 4)
	ax.plot(wl*1e4, spe, linewidth = 1, label = 'Composite spectrum', color = 'seagreen', zorder=3.5)
	ax.plot(wl*1e4, pri_spec, linewidth = 1, label = 'Primary: {:.0f}K'.format(tt1), color = 'darkblue', zorder = 3)
	ax.plot(wl*1e4, sec_spec, linewidth = 1, label = 'Secondary: {:.0f}K'.format(tt2), color = 'darkorange', zorder = 3)
	if tell == True:
		[ax.axvspan(*r, alpha=0.3, color='0.4', zorder = 5) for r in regions]

	for n in range(len(random_sample)):
		# if n == 0:
		if len(a) == 6 and dist_fit == True:
			t1, t2, e, r1, r2, pl = random_sample[n]
			random_lg = [get_logg(t, matrix) for t in [t1, t2]]
			dist = 1/pl

			ww, sspe, ppri_spec, ssec_spec, ppri_mag, ssec_mag, gpm, gsm, gm = make_composite([t1, t2], random_lg, [r1, r2], pl, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		elif len(a) == 6 and dist_fit == False:
			t1, t2, e, r1, r2, pl = random_sample[n]
			random_lg = [get_logg(t, matrix) for t in [t1, t2]]
			ratio1 = r2
			ww, sspe, ppri_spec, ssec_spec, ppri_mag, ssec_mag, gpm, gsm, gm = make_composite([t1, t2], random_lg, [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)

		else:
			t1, t2, e, ratio1 = random_sample[n]
			random_lg = [get_logg(t, matrix) for t in [t1, t2]]

			ww, sspe, ppri_spec, ssec_spec, ppri_mag, ssec_mag, gpm, gsm, gm = make_composite([t1, t2], random_lg, [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)

		sspe = extinct(ww, sspe, e)
		ppri_spec = extinct(ww, ppri_spec, e)
		ssec_spec = extinct(ww, ssec_spec, e)

		ite = interp1d(ww, sspe)
		sspe = ite(wl*1e4)

		it1, it2 = interp1d(ww, ppri_spec), interp1d(ww, ssec_spec)
		ppri_spec, ssec_spec = it1(wl*1e4), it2(wl*1e4)

		if len(a) == 6:
			ratio = rad2

		ppri_spec *= pri_ratio; ssec_spec *= sec_ratio

		sspe *= np.median(spec)/np.median(sspe)

		ax.plot(wl*1e4, sspe, linewidth = 0.75, color = 'limegreen', alpha = 0.5, zorder = 2.5, rasterized = True)
		ax.plot(wl*1e4, ppri_spec, linewidth = 0.75, color = 'skyblue', alpha = 0.5, zorder = 2, rasterized = True)
		ax.plot(wl*1e4, ssec_spec, linewidth = 0.75, color = 'gold', alpha = 0.5, zorder = 2, rasterized = True)

		ax1.plot(wl*1e4, spec - sspe, linewidth = 0.5, color = '0.7', alpha = 0.5, zorder = 0, rasterized = True)

	plt.minorticks_on()
	ax.set_rasterization_zorder(0)
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = 14, direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	# ax.set_xlim(6900, 7500)

	ax1.plot(wl*1e4, spec - spe, linewidth = 1, color = 'k', label = 'Data - composite', zorder = 2)
	if tell == True:
		[ax1.axvspan(*r, alpha=0.3, color='0.4', zorder = 5) for r in regions]
	ax1.axhline(0, label = 'No resid.', linestyle = '--', color ='k', linewidth = 1, zorder = 1)
	ax1.legend(loc = 'best', fontsize = 10, ncol = 2)
	ax1.tick_params(which='both', labelsize = 14, direction='in')

	ax1.set_xlabel(r'Wavelength ($\AA$)', fontsize = 14)
	ax.set_ylabel('{}'.format(r'Normalized Flux'), fontsize = 14)
	ax1.set_ylabel('Resid.', fontsize = 14)
	ax.legend(loc = 'best', fontsize = 12)
	plt.tight_layout()
	plt.savefig(run + '/plots/{}_all_spec.pdf'.format(fname))

	############
	#CREATE THE COMPOSITE + DATA PLOT 
	###########

	if not real_val[0] == 0:
		if len(real_val) == 6 and dist_fit == True:
			rt1, rt2, rr1, rr2, rpl = real_val
			log_g_val = [get_logg(t, matrix) for t in [rt1, rt2]]
			real_wl, rspec, rpspec, rsspec, pri_mag, sec_mag, gpm, gsm, gm = make_composite([rt1, rt2], log_g_val, [rr1, rr2], rpl, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		elif len(real_val) == 6 and dist_fit == False:
			rt1, rt2, rr1, rr2, rpl = real_val
			log_g_val = [get_logg(t, matrix) for t in [rt1, rt2]]
			ratio = rr2
			real_wl, rspec, rpspec, rsspec, pri_mag, sec_mag, gpm, gsm, gm = make_composite([rt1, rt2], log_g_val, [ratio], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		else:
			rt1, rt2, ratio = real_val
			log_g_val = [get_logg(t, matrix) for t in [rt1, rt2]]
			real_wl, rspec, rpspec, rsspec, pri_mag, sec_mag, gpm, gsm, gm = make_composite([rt1, rt2], log_g_val, [ratio], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		# rspec = extinct(real_wl, rspec, re)
		rspec *= np.median(spec)/np.median(rspec[np.where((real_wl < max(wl * 1e4)) & (real_wl > min(wl * 1e4)))])

	fig, ax = plt.subplots()
	ax.plot(wl*1e4, spec, linewidth = 1, label = 'Data spectrum', color = 'navy', zorder = 0)
	ax.plot(wl*1e4, spe, linewidth = 1, label = 'Model: {:.0f}K + {:.0f}K'.format(tt1, tt2), color = 'xkcd:sky blue', zorder=1)
	if not real_val[0] == 0:
		ax.plot(real_wl, rspec, linewidth = 1, color = 'xkcd:grass green', label = 'B15 values: {:.0f}K + {:.0f}K'.format(rt1, rt2))
	# ax.set_xlim(max(min(w), min(wl*1e4)), min(max(w), max(wl)*1e4))
	ax.set_xlim(8500, 8700)
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	plt.xlabel(r'Wavelength (\AA)', fontsize = 13)
	plt.ylabel('Normalized flux', fontsize = 13)
	plt.legend(loc = 'best', fontsize = 13)
	plt.tight_layout()
	plt.savefig(run + '/plots/bestfit_spec_post_mcmc.pdf')
	plt.close()


	############
	#COMPUTE AND CREATE THE KEPLER CONTRAST AND CORRECTION FACTOR PLOTS
	###########

	kep_sample = sample[np.random.choice(len(sample), size = int(2000), replace = False), :]
	kep_contrast = []; kep_rad = []; gaia_contrast = []; gaiamag_all = []
	for n in range(len(kep_sample)):
		if len(a) == 6 and dist_fit == True:
			tt1, tt2, ex, rad1, rad2, plx = kep_sample[n,:]
			random_lg = [get_logg(t, matrix) for t in [tt1, tt2]]
			ratio1 = rad2
			w, spe, pri_spec, sec_spec, pri_mag, sec_mag, pri_mag_gaia, sec_mag_gaia, gaiamag = make_composite([tt1, tt2], random_lg, [rad1, rad2], plx, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		elif len(a) == 6 and dist_fit == False:
			tt1, tt2, ex, rad1, rad2, plx = kep_sample[n,:]
			random_lg = [get_logg(t, matrix) for t in [tt1, tt2]]
			ratio1 = rad2
			w, spe, pri_spec, sec_spec, pri_mag, sec_mag, pri_mag_gaia, sec_mag_gaia, gaiamag = make_composite([tt1, tt2], random_lg, [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		else:	
			tt1, tt2, te, ratio1 = kep_sample[n,:]
			random_lg = [get_logg(t, matrix) for t in [tt1, tt2]]
			w, spe, pri_spec, sec_spec, pri_mag, sec_mag, pri_mag_gaia, sec_mag_gaia, gaiamag = make_composite([tt1, tt2], random_lg, [ratio1], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		
		kep_rad.append(ratio1)
		kep_contrast.append(sec_mag-pri_mag)
		gaia_contrast.append(sec_mag_gaia)
		gaiamag_all.append(pri_mag_gaia)

	kep_contrast = np.array(kep_contrast); kep_rad = np.array(kep_rad)
	gaiamag_all = np.array(gaiamag_all); gaia_contrast = np.array(gaia_contrast) 

	gaiamag_all = gaiamag_all #+ ex *0.789

	nbins = 110
	contrast_bins = np.linspace(min(kep_contrast), max(kep_contrast), nbins)
	contrast_count = np.zeros(len(contrast_bins))


	for t in kep_contrast:
		for b in range(len(contrast_bins) - 1):
			if contrast_bins[b] <= t < contrast_bins[b + 1]:
				contrast_count[b] += 1

	kep_mean = np.abs(np.percentile(np.array(kep_contrast), 50))
	kep_84 = np.abs(np.percentile(np.array(kep_contrast), 84))
	kep_16 = np.abs(np.percentile(np.array(kep_contrast),16))

	fig, ax = plt.subplots(figsize = (4, 4))
	ax.hist(kep_contrast, histtype = 'step', linewidth = 2, color = 'k')
	ax.axvline(kep_84, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(kep_16, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(kep_mean, linestyle = '-', color = 'k', linewidth = 2)
	ax.set_title(r'$\Delta$Kep = {:.3f}$^{{+ {:.3f}}}_{{- {:.3f}}}$'.format(kep_mean, kep_84-kep_mean, kep_mean - kep_16))
	ax.set_xlabel(r'$\Delta$Kep (mag)')
	plt.tight_layout()
	plt.savefig(run + '/plots/{}_delta_kep.pdf'.format(fname))
	plt.close()

	np.savetxt(run + '/kep_contrast.txt', np.array(kep_contrast))
	np.savetxt(run + '/gaia_sec.txt', np.array(gaia_contrast))
	np.savetxt(run + '/gaia_pri.txt', gaiamag_all)


	pri_corr = np.sqrt(1 + 10**(-0.4 * kep_contrast)) #from Furlan+2017; this is assuming the primary radius is equal to the kepler radius
	sec_corr = kep_rad * np.sqrt(1 + 10**(0.4 * kep_contrast) * pri_corr**2)

	pc_bins = np.linspace(min(pri_corr), max(pri_corr), nbins)
	sc_bins = np.linspace(min(sec_corr), max(sec_corr), nbins)
	pc_count = np.zeros(len(pc_bins))
	sc_count = np.zeros(len(sc_bins))
	for t in pri_corr:
		for b in range(len(pc_bins) - 1):
			if pc_bins[b] <= t < pc_bins[b + 1]:
				pc_count[b] += 1

	for t in sec_corr:
		for b in range(len(sc_bins) - 1):
			if sc_bins[b] <= t < sc_bins[b + 1]:
				sc_count[b] += 1

	pri_mean = np.abs(np.percentile(np.array(pri_corr), 50))
	pri_84 = np.abs(np.percentile(np.array(pri_corr), 84))
	pri_16 = np.abs(np.percentile(np.array(pri_corr),16))

	fig, ax = plt.subplots(figsize = (4, 4))
	ax.hist(pri_corr, histtype = 'step', linewidth = 2, color = 'k')
	ax.axvline(pri_84, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(pri_16, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(pri_mean, linestyle = '-', color = 'k', linewidth = 2)
	ax.set_title(r'$f_{{p, corr}}$ = {:.4f}$^{{+ {:.4f}}}_{{- {:.4f}}}$'.format(pri_mean, pri_84-pri_mean, pri_mean - pri_16))
	ax.set_xlabel(r'Corr. factor (primary)')
	plt.tight_layout()
	plt.savefig(run + '/plots/{}_pri_corr.pdf'.format(fname))
	plt.close()

	np.savetxt(run + '/pri_corr.txt', np.array(pri_corr))

	sec_mean = np.abs(np.percentile(np.array(sec_corr), 50))
	sec_84 = np.abs(np.percentile(np.array(sec_corr), 84))
	sec_16 = np.abs(np.percentile(np.array(sec_corr),16))

	fig, ax = plt.subplots(figsize = (4, 4))
	ax.hist(sec_corr, histtype = 'step', linewidth = 2, color = 'k')
	ax.axvline(sec_84, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(sec_16, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(sec_mean, linestyle = '-', color = 'k', linewidth = 2)
	ax.set_title(r'$f_{{s, corr}}$ = {:.3f}$^{{+ {:.3f}}}_{{- {:.3f}}}$'.format(sec_mean, sec_84-sec_mean, sec_mean - sec_16))
	ax.set_xlabel(r'Corr. factor (secondary)')
	plt.tight_layout()
	plt.savefig(run + '/plots/{}_sec_corr.pdf'.format(fname))
	plt.close()

	np.savetxt(run + '/sec_corr.txt', np.array(sec_corr))

	############
	#CREATE ISOCHRONE + COMPONENT PLOT
	###########

	#make this in H-K vs. K space so I can use contrasts to get a mag for each component
	matrix = np.genfromtxt('mist_2mass_old.cmd', autostrip = True)
	#get the age and convert it into Gyr from log(years)
	aage = matrix[:, 1]

	teff5, lum5 = matrix[:,4][np.where(np.array(aage) == 9.0000000000000000)], matrix[:,6][np.where(np.array(aage) == 9.0000000000000000)]
	ma5 = matrix[:,3][np.where(np.array(aage) == 9.0000000000000000)]

	aage = np.array([(10**a)/1e9 for a in aage])

	#get the mass, log(luminosity), effective temp, and log(g)
	ma = matrix[:,3][np.where((aage > 0.1) & (aage < 8))]
	teff = matrix[:, 4][np.where((aage > 0.1) & (aage < 8))]
	lum = matrix[:, 6][np.where((aage > 0.1) & (aage < 8))]
	hmag = matrix[:, 15][np.where((aage > 0.1) & (aage < 8))]
	kmag = matrix[:, 16][np.where((aage > 0.1) & (aage < 8))]
	hk_color = np.array(hmag) - np.array(kmag)
	aage = aage[np.where((aage > 0.1) & (aage < 8))]

	lum, lum5 = [10**l for l in lum], [10**l for l in lum5]; teff, teff5 = [10**t for t in teff], [10**t for t in teff5]

	#remove redundant ages from the age vector
	a1 = [aage[0]]
	for n in range(len(aage)):
		if aage[n] != a1[-1]:
			a1.append(aage[n])

	#plot the age = 0 temperature vs. luminosity 
	fig, ax = plt.subplots()

	#now for all the other ages fill an array with the single valued age, get the temperature and convert it from log
	#then plot it versus the correct luminosity
	#tagging each one with the age and color coding it 
	for n in np.arange(0, len(a1), 4):
		a2 = np.full(len(np.where(aage == a1[n])[0]), a1[n])

		if n == 0:
			ax.plot(np.array(teff)[np.where(np.array(aage) == a1[n])], np.log10(np.array(lum)[np.where(np.array(aage) == a1[n])]), color = cm.plasma(a1[n]/10), zorder = 0, label = 'MS')#, label = '{}'.format(int(np.around(a1[n]))))
		else:
			ax.plot(np.array(teff)[np.where(np.array(aage) == a1[n])], np.log10(np.array(lum)[np.where(np.array(aage) == a1[n])]), color = cm.plasma(a1[n]/10), zorder = 0)#, label = '{}'.format(int(np.around(a1[n]))))

	#calculate the intrinsic H and K mags from the SED to get the color
	if len(a) == 6 and dist_fit == True:
		tt1, tt2, ex, rad1, rad2, plx = a
		ratio1 = rad2
	elif len(a) == 6 and dist_fit == False:
		tt1, tt2, ex, rad1, rad2, plx = a
		ratio1 = rad2
	else:	
		tt1, tt2, tl1, tl2, te, ratio1 = a

	l_intep = interp1d(teff5[:200], lum5[:200]); pri_lum = l_intep(tt1)

	sigma_sb = 5.670374e-5 #erg/s/cm^2/K^4
	lsun = 3.839e33 #erg/s 
	rsun = 6.955e10
	pri_rad = np.sqrt(pri_lum*lsun/(4 * np.pi * sigma_sb * tt1**4)) #cm

	sec_rad = ratio1 * pri_rad
	sec_lum = (4 * np.pi * sec_rad**2 * sigma_sb * tt2**4)/lsun #solar luminosity

	ax.scatter(tt1, np.log10(pri_lum), marker = 'x', color = 'darkgray', s = 60, label = 'Primary')
	ax.scatter(tt2, np.log10(sec_lum), marker = 'x', color = 'darkorange', s = 50, label = 'Secondary')
	ax.set_xlabel(r'T$_{eff}$ (K)', fontsize = 16)
	ax.set_ylabel(r'$\log_{10}$(L (L$_{\odot}$))', fontsize = 16)
	# ax.set_yscale('log')
	ax.set_xlim(5000, 3000)
	ax.set_ylim(np.log10(1e-3), np.log10(1))
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = 16, direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.legend(loc = 'best', fontsize = 13)
	fig.tight_layout()
	#save the figure in the run directory
	plt.savefig(run + '/plots/{}_isochrone.pdf'.format(fname))
	plt.close()

	ma_intep = interp1d(teff5[:200], ma5[:200])

	pmass_posterior = []; plum_posterior = []
	smass_posterior = []; slum_posterior = []
	for n in range(len(kep_sample)):
		tt1, tt2, ex, rad1, rad2, plx = kep_sample[n,:]
		interp_map = ma_intep(tt1)
		interp_mas = ma_intep(tt2)
		pmass_posterior.append(float(interp_map)); smass_posterior.append(float(interp_mas))

		interp_plum = l_intep(tt1)
		interp_slum = l_intep(tt2)
		plum_posterior.append(float(interp_plum)); slum_posterior.append(float(interp_slum))


	savefig_titles = ['primary_mass_posterior', 'secondary_mass_posterior', 'primary_lum_posterior', 'secondary_lum_posterior']
	figure_titles = ['M$_{{pri}}$', 'M$_{{sec}}$', 'L$_{{pri}}$', 'L$_{{sec}}$']
	xax_label = ['M$_{{pri}} (M_{{\odot}})$', 'M$_{{sec}} (M_{{\odot}})$', 'L$_{{pri}} (L_{{\odot}})$', 'L$_{{sec}} (L_{{\odot}})$']
	for n, posterior in enumerate([pmass_posterior, smass_posterior, plum_posterior, slum_posterior]):

		sc_bins = np.linspace(min(posterior), max(posterior), nbins)
		sc_count = np.zeros(len(sc_bins))

		for t in posterior:
			for b in range(len(sc_bins) - 1):
				if sc_bins[b] <= t < sc_bins[b + 1]:
					sc_count[b] += 1
		sec_mean = np.abs(np.percentile(np.array(posterior), 50))
		sec_84 = np.abs(np.percentile(np.array(posterior), 84))
		sec_16 = np.abs(np.percentile(np.array(posterior),16))

		fig, ax = plt.subplots(figsize = (4, 4))
		ax.hist(posterior, histtype = 'step', linewidth = 2, color = 'k')
		ax.axvline(sec_84, linestyle = '--', color = 'k', linewidth = 2)
		ax.axvline(sec_16, linestyle = '--', color = 'k', linewidth = 2)
		ax.axvline(sec_mean, linestyle = '-', color = 'k', linewidth = 2)
		ax.set_title(r'{} = {:.3f}$^{{+ {:.3f}}}_{{- {:.3f}}}$'.format(figure_titles[n], sec_mean, sec_84-sec_mean, sec_mean - sec_16))
		ax.set_xlabel(r'{}'.format(xax_label[n]))
		plt.tight_layout()
		plt.savefig(run + '/plots/{}.pdf'.format(savefig_titles[n]))
		plt.close()

		np.savetxt(run + '/' + savefig_titles[n] + '.txt', np.array(posterior))

	return

def plot_results3(fname, sample, run, data, sp, fr, ctm, ptm, tmi, tma, vs, real_val, distance, matrix, models = 'btsettl', res = 1000, dist_fit = True, tell = False):
	#unpack the sample array
	t1, t2, t3, av, r1, r2, r3, pl = sample.T

	#calculate the median value for all the emcee chains
	a = np.median(sample, axis = 0)
	#draw a random sample of parameters from the emcee draws 
	random_sample = sample[np.random.choice(len(sample), size = 100, replace = False), :]
	random_sample = random_sample[np.where((random_sample[:,2] > 2900) & (random_sample[:,1] > 2900) & (random_sample[:,1] < 6200) & (random_sample[:,0] < 6200) & (random_sample[:,2] < 6200))]


	############
	#FIND AND PLOT THE BEST-FIT PARAMETERS IN THE BIMODAL DISTRIBUTIONS
	###########
	#define the number of bins to use to create the PDFs
	nbins = 75

	#initialize some arrays 
	#*bins defines the histogram bins; *count is the PDF for the bins (non-normalized)
	t1_bins, t2_bins, t3_bins = np.linspace(min(t1), max(t1), nbins), np.linspace(min(t2), max(t2), nbins), np.linspace(min(t3), max(t3), nbins)
	r1_bins, r2_bins, r3_bins = np.linspace(min(r1), max(r1), nbins), np.linspace(min(r2), max(r2), nbins), np.linspace(min(r3), max(r3), nbins)

	t1_count, t2_count = np.zeros(len(t1_bins)), np.zeros(len(t2_bins)); t3_count = np.zeros(len(t1_bins))
	r1_count, r2_count = np.zeros(len(r1_bins)), np.zeros(len(r2_bins)); r3_count = np.zeros(len(t1_bins))

	#fill up the count bins 
	#temperature 1
	for t in t1:
		for b in range(len(t1_bins) - 1):
			if t1_bins[b] <= t < t1_bins[b + 1]:
				t1_count[b] += 1

	#Teff 2
	for t in t2:
		for b in range(len(t2_bins) - 1):
			if t2_bins[b] <= t < t2_bins[b + 1]:
				t2_count[b] += 1

	#Teff 3
	for t in t3:
		for b in range(len(t3_bins) - 1):
			if t3_bins[b] <= t < t3_bins[b + 1]:
				t3_count[b] += 1

	#radius 1
	for r in r1:
		for b in range(len(r1_bins) - 1):
			if r1_bins[b] <= r < r1_bins[b + 1]:
				r1_count[b] += 1

	#Radius 2
	for r in r2:
		for b in range(len(r2_bins) - 1):
			if r2_bins[b] <= r < r2_bins[b + 1]:
				r2_count[b] += 1

	#Radius 3
	for r in r3:
		for b in range(len(r3_bins) - 1):
			if r3_bins[b] <= r < r3_bins[b + 1]:
				r3_count[b] += 1

	#####
	#Temperature 1
	#####
	try:
		#find the local min between the two gaussian temperature measurements 
		t1_localmin = int(np.mean([np.where(t1_count[np.where((t1_bins > min(t1)) & (t1_bins < max(t1)))] < 0.5*max(t1_count))]))
		#fit a bimodal gaussian to the distribution
		fit_t1, cov = curve_fit(bimodal, t1_bins, t1_count, [np.mean(t1_bins[t1_localmin:]), np.std(t1_bins[t1_localmin:]), max(t1_count[t1_localmin:]),\
			 np.mean(t1_bins[:t1_localmin]), np.std(t1_bins[:t1_localmin]), max(t1_count[:t1_localmin])])

		#plot the bimodal distribution, the best fit, and the corresponding component Gaussians
		plt.figure()
		plt.hist(t1, bins = t1_bins)
		plt.axvline(t1_bins[t1_localmin], color = 'k', linewidth = 2)
		plt.plot(t1_bins, t1_count)
		plt.plot(t1_bins, bimodal(t1_bins, *fit_t1), color = 'b')
		plt.plot(t1_bins, gauss(t1_bins, *fit_t1[:3]))
		plt.plot(t1_bins, gauss(t1_bins, *fit_t1[3:]))
		plt.savefig(run + '/plots/bimodal_test_T1.pdf')

		#calculate the area under each curve as a percentage
		t1_v1 = np.trapz(gauss(t1_bins, *fit_t1[:3]))/np.trapz(bimodal(t1_bins, *fit_t1))
		t1_v2 = np.trapz(gauss(t1_bins, *fit_t1[3:]))/np.trapz(bimodal(t1_bins, *fit_t1))

		#pick the gaussian with the largest area and set the appropriate parameters to the corresponding values as established from the fit
		if t1_v1 > t1_v2:
			a[0] = fit_t1[0]; sigma_t1 = fit_t1[1]; 
		else:
			a[0] = fit_t1[3]; sigma_t1 = fit_t1[4]
		p_t1 = max(t1_v1, t1_v2)
	#if none of that works because it's not bimodal, just set the stddev to 0 and the best-fit value will stay at the median value from emcee	
	except:
		sigma_t1 = 0
		pass;

	#####
	#Temperature 2
	#####
	try:
		t2_localmin = int(np.mean([np.where(t2_count[np.where((t2_bins > min(t2)) & (t2_bins < max(t2)))] < 0.5*max(t2_count))]))

		fit_t2, cov = curve_fit(bimodal,t2_bins, t2_count, [np.mean(t2_bins[t2_localmin:]), np.std(t2_bins[t2_localmin:]), max(t2_count[t2_localmin:]), \
				np.mean(t2_bins[:t2_localmin]), np.std(t2_bins[:t2_localmin]), max(t2_count[:t2_localmin])])

		plt.figure()
		plt.hist(t2, bins = t2_bins)
		plt.axvline(t2_bins[t2_localmin], color = 'k', linewidth = 2)
		plt.plot(t2_bins, t2_count)
		plt.plot(t2_bins, bimodal(t2_bins, *fit_t2), color = 'b')
		plt.plot(t2_bins, gauss(t2_bins, *fit_t2[:3]))
		plt.plot(t2_bins, gauss(t2_bins, *fit_t2[3:]))
		plt.savefig(run + '/plots/bimodal_test_T2.pdf')


		t2_v1 = np.trapz(gauss(t2_bins, *fit_t2[:3]))/np.trapz(bimodal(t2_bins, *fit_t2))
		t2_v2 = np.trapz(gauss(t2_bins, *fit_t2[3:]))/np.trapz(bimodal(t2_bins, *fit_t2))
		if t2_v1 > t2_v2:
			a[1] = fit_t2[0]; sigma_t2 = fit_t2[1]; 
		else:
			a[1] = fit_t2[3]; sigma_t2 = fit_t2[4]
		p_t2 = max(t2_v1, t2_v2)
	except:
		sigma_t2 = 0
		pass;

	try:
		t3_localmin = int(np.mean([np.where(t3_count[np.where((t3_bins > min(t3)) & (t3_bins < max(t3)))] < 0.5*max(t3_count))]))

		fit_t3, cov = curve_fit(bimodal,t3_bins, t3_count, [np.mean(t3_bins[t3_localmin:]), np.std(t3_bins[t3_localmin:]), max(t3_count[t3_localmin:]), \
				np.mean(t3_bins[:t3_localmin]), np.std(t3_bins[:t3_localmin]), max(t3_count[:t3_localmin])])

		plt.figure()
		plt.hist(t3, bins = t3_bins)
		plt.axvline(t3_bins[t3_localmin], color = 'k', linewidth = 2)
		plt.plot(t3_bins, t3_count)
		plt.plot(t3_bins, bimodal(t3_bins, *fit_t3), color = 'b')
		plt.plot(t3_bins, gauss(t3_bins, *fit_t3[:3]))
		plt.plot(t3_bins, gauss(t3_bins, *fit_t3[3:]))
		plt.savefig(run + '/plots/bimodal_test_T3.pdf')


		t3_v1 = np.trapz(gauss(t3_bins, *fit_t3[:3]))/np.trapz(bimodal(t3_bins, *fit_t3))
		t3_v2 = np.trapz(gauss(t3_bins, *fit_t3[3:]))/np.trapz(bimodal(t3_bins, *fit_t3))
		if t3_v1 > t3_v2:
			a[2] = fit_t3[0]; sigma_t3 = fit_t3[1]; 
		else:
			a[2] = fit_t3[3]; sigma_t3 = fit_t3[4]
		p_t3 = max(t3_v1, t3_v2)
	except:
		sigma_t3 = 0
		pass;

	#####
	#Radius 1
	#####
	try:
		r1_localmin = int(np.mean([np.where(r1_count[np.where((r1_bins > min(r1)) & (r1_bins < max(r1)))] < 0.5*max(r1_count))]))
		fit_r1, cov = curve_fit(bimodal, r1_bins, r1_count, [np.mean(r1_bins[r1_localmin:]), np.std(r1_bins[r1_localmin:]), max(r1_count[r1_localmin:]),\
			 np.mean(r1_bins[:r1_localmin]), np.std(r1_bins[:r1_localmin]), max(r1_count[:r1_localmin])])

		plt.figure()
		plt.hist(r1, bins = r1_bins)
		plt.axvline(r1_bins[r1_localmin], color = 'k', linewidth = 2)
		plt.plot(r1_bins, r1_count)
		plt.plot(r1_bins, bimodal(r1_bins, *fit_r1), color = 'b')
		plt.plot(r1_bins, gauss(r1_bins, *fit_r1[:3]))
		plt.plot(r1_bins, gauss(r1_bins, *fit_r1[3:]))
		plt.savefig(run + '/plots/bimodal_test_R1.pdf')

		r1_v1 = np.trapz(gauss(r1_bins, *fit_r1[:3]))/np.trapz(bimodal(r1_bins, *fit_r1))
		r1_v2 = np.trapz(gauss(r1_bins, *fit_r1[3:]))/np.trapz(bimodal(r1_bins, *fit_r1))
		if r1_v1 > r1_v2:
			a[7] = fit_r1[0]; sigma_r1 = fit_r1[1]; 
		else:
			a[7] = fit_r1[3]; sigma_r1 = fit_r1[4]
		p_r1 = max(r1_v1, r1_v2)
	except:
		sigma_r1 = 0
		pass;

	#####
	#Radius ratio 1
	#####
	try:
		r2_localmin = int(np.mean([np.where(r2_count[np.where((r2_bins > min(r2)) & (r2_bins < max(r2)))] < 0.5*max(r2_count))]))
		fit_r2, cov = curve_fit(bimodal,r2_bins, r2_count, [np.mean(r2_bins[r2_localmin:]), np.std(r2_bins[r2_localmin:]), max(r2_count[r2_localmin:]), \
				np.mean(r2_bins[:r2_localmin]), np.std(r2_bins[:r2_localmin]), max(r2_count[:r2_localmin])])

		plt.figure()
		plt.hist(r2, bins = r2_bins)
		plt.axvline(r2_bins[r2_localmin], color = 'k', linewidth = 2)
		plt.plot(r2_bins, r2_count)
		plt.plot(r2_bins, bimodal(r2_bins, *fit_r2), color = 'b')
		plt.plot(r2_bins, gauss(r2_bins, *fit_r2[:3]))
		plt.plot(r2_bins, gauss(r2_bins, *fit_r2[3:]))
		plt.savefig(run + '/plots/bimodal_test_R2.pdf')

		r2_v1 = np.trapz(gauss(r2_bins, *fit_r2[:3]))/np.trapz(bimodal(r2_bins, *fit_r2))
		r2_v2 = np.trapz(gauss(r2_bins, *fit_r2[3:]))/np.trapz(bimodal(r2_bins, *fit_r2))
		if r2_v1 > r2_v2:
			a[8] = fit_r2[0]; sigma_r2 = fit_r2[1]; 
		else:
			a[8] = fit_r2[3]; sigma_r2 = fit_r2[4]
		p_r2 = max(r2_v1, r2_v2)
	except:
		sigma_r2 = 0
		pass;

	#####
	#Radius ratio 2 
	#####
	try:
		r3_localmin = int(np.mean([np.where(r3_count[np.where((r3_bins > min(r3)) & (r3_bins < max(r3)))] < 0.5*max(r3_count))]))
		fit_r3, cov = curve_fit(bimodal,r3_bins, r3_count, [np.mean(r3_bins[r3_localmin:]), np.std(r3_bins[r3_localmin:]), max(r3_count[r3_localmin:]), \
				np.mean(r3_bins[:r3_localmin]), np.std(r3_bins[:r3_localmin]), max(r3_count[:r3_localmin])])

		plt.figure()
		plt.hist(r3, bins = r3_bins)
		plt.axvline(r3_bins[r3_localmin], color = 'k', linewidth = 2)
		plt.plot(r3_bins, r3_count)
		plt.plot(r3_bins, bimodal(r3_bins, *fit_r3), color = 'b')
		plt.plot(r3_bins, gauss(r3_bins, *fit_r3[:3]))
		plt.plot(r3_bins, gauss(r3_bins, *fit_r3[3:]))
		plt.savefig(run + '/plots/bimodal_test_R3.pdf')

		r3_v1 = np.trapz(gauss(r3_bins, *fit_r3[:3]))/np.trapz(bimodal(r3_bins, *fit_r3))
		r3_v2 = np.trapz(gauss(r3_bins, *fit_r3[3:]))/np.trapz(bimodal(r3_bins, *fit_r3))
		if r3_v1 > r3_v2:
			a[9] = fit_r3[0]; sigma_r3 = fit_r3[1]; 
		else:
			a[9] = fit_r3[3]; sigma_r3 = fit_r3[4]
		p_r3 = max(r3_v1, r3_v2)
	except:
		sigma_r3 = 0
		pass;


	############
	#CREATE THE PHOTOMETRY AND CONTRAST PLOTS
	###########

	plt.minorticks_on()
	wl, spec, err = data
	#unpack the best-fit stellar parameters 
	if len(a) == 8:
		tt1, tt2, tt3, te, rad1, rad2, rad3, plx = np.median(sample, axis = 0)
		lg_guess = [get_logg(t, matrix) for t in [tt1, tt2, tt3]]

		ww, ss, c, pcwl, phot = make_composite([tt1, tt2, tt3], lg_guess, [rad1, rad2, rad3], plx, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = False, nspec = 3)
		
		w, spe, pri_spec, sec_spec, tri_spec, pri_mag, sec_mag, tri_mag = make_composite([tt1, tt2, tt3], lg_guess, [rad1, rad2, rad3], plx, fr[2], fr[5], [5000, 24000], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True, nspec = 3)
		new_wl, new_prispec = redres(w, pri_spec, 250)
		new_wl, new_spe = redres(w, spe, 250)
		new_wl, new_secspec = redres(w, sec_spec, 250)
		new_wl, new_trispec = redres(w, tri_spec, 250)

		zp = [2.854074834606756e-09,1.940259205607388e-09,1.359859453789013e-09,3.1121838042516567e-10,1.1353317746392182e-10, 4.279017715611946e-11]
		phot_cwl = [6175, 7489, 8946, 12350, 16620, 21590]
		phot_width = np.array([[6175-5415, 6989-6175], [7489-6689, 8389-7489], [8946-7960, 10833-8946], [12350-10806, 14067-12350], [16620-14787, 18231-16620], [21590-19543, 23552-21590]]).T

		fig,ax = plt.subplots(nrows = 3, gridspec_kw = dict(hspace = 0, height_ratios = [3, 1.75, 1]), sharex = True, figsize = (7 , 6))
		e = ax[0].scatter(phot_cwl, 10**(-0.4*phot)*zp[:len(phot)], color = 'seagreen', s = 100, marker = '.', label = 'Composite phot.')
		ax[0].errorbar(phot_cwl, 10**(-0.4*phot)*zp[:len(phot)], xerr = phot_width, color= 'seagreen', zorder = 0, linestyle = 'None')
		b = ax[0].scatter(phot_cwl, 10**(-0.4*fr[3])*zp[:len(phot)], linestyle = 'None', color = 'k', marker = '.', s = 100, label = 'Data phot.')
		m = ax[0].plot(new_wl, new_spe, color = 'seagreen', linewidth = 1, zorder = 0, alpha = 0.5)
		plt.minorticks_on()
		ax[0].set_xscale('log')
		ax[0].set_yscale('log')
		ax[0].tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		ax[0].tick_params(bottom=True, top =True, left=True, right=True)
		ax[0].tick_params(which='both', labelsize = 12, direction='in')
		ax[0].tick_params('both', length=8, width=1.5, which='major')
		ax[0].tick_params('both', length=4, width=1, which='minor')
		ax[0].set_ylabel('{}'.format(r'Flux (erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)'), fontsize = 12)

		f = ax[1].scatter(ctm[3][:int(len(c)/2)], c[:int(len(c)/2)], color = 'blue', marker = 'v', label = 'Sec. contrast', zorder = 2)
		m = ax[1].scatter(ctm[3][int(len(c)/2):], c[int(len(c)/2):], color = 'gold', marker = 'v', label = 'Tri. contrast', zorder = 2)
		ax[1].errorbar(ctm[3], fr[0], yerr = fr[1], linestyle = 'None', capsize = 4, capthick = 2, color = 'k', marker = 'v', zorder = 1)
		g = ax[1].scatter(ctm[3], fr[0], color = 'k', marker = 'v', label = 'Data contrast', zorder = 1)
		ax[1].plot(new_wl, 2.5*np.log10(new_prispec) - 2.5*np.log10(new_secspec), color = 'blue', linewidth = 1, zorder = 0, alpha = 0.5)
		ax[1].plot(new_wl, 2.5*np.log10(new_prispec) - 2.5*np.log10(new_trispec), color = 'gold', linewidth = 1, zorder = 0, alpha = 0.5)
		ax[1].set_ylabel(r'$\Delta$ mag', fontsize = 12)

		plt.minorticks_on()
		ax[1].tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		ax[1].tick_params(bottom=True, top =True, left=True, right=True)
		ax[1].tick_params(which='both', labelsize = 12, direction='in')
		ax[1].tick_params('both', length=8, width=1.5, which='major')
		ax[1].tick_params('both', length=4, width=1, which='minor')

		ax[2].scatter(phot_cwl, phot-fr[3], color = 'seagreen', marker = 'x', s = 50, label = 'Phot. resid.')
		ax[2].axhline(0, color = '0.3', linestyle = '--', linewidth = 2, label = 'No resid.')
		ax[2].scatter(ctm[3][:int(len(c)/2)], np.array(fr[0][:int(len(c)/2)])-np.array(c[:int(len(c)/2)]), color = 'blue', marker = 'x', label = 'Cont. resid.',s = 50)
		ax[2].scatter(ctm[3][int(len(c)/2):], np.array(fr[0][int(len(c)/2):])-np.array(c[int(len(c)/2):]), color = 'gold', marker = 'x', label = 'Cont. resid.',s = 50)
		plt.minorticks_on()
		ax[2].tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		ax[2].tick_params(bottom=True, top =True, left=True, right=True)
		ax[2].tick_params(which='both', labelsize = 12, direction='in')
		ax[2].tick_params('both', length=8, width=1.5, which='major')
		ax[2].tick_params('both', length=4, width=1, which='minor')
		ax[2].set_xlabel(r'Wavelength (\AA)', fontsize = 12)
		ax[2].set_ylabel('{}'.format(r'Resid. (mag)'), fontsize = 12)

		fig.align_ylabels(ax)

		handles, labels = plt.gca().get_legend_handles_labels()
		handles.extend([e,b,f,g])

		ax[0].legend(handles = handles, loc = 'best', fontsize = 10, ncol = 2)

		plt.tight_layout()
		plt.savefig(run + '/plots/{}_phot_scatter.pdf'.format(fname))
		plt.close()

	############
	#CREATE THE COMPOSITE + COMPONENET + DATA PLOT (all_spec)
	###########

	#contrast and phot (unresolved photometry) in flux
	#pwl is the central wavelength for the photometric filters
	if len(a) == 8 and dist_fit == True:
		w, spe, pri_spec, sec_spec, tri_spec, pri_mag, sec_mag, tri_mag = make_composite([tt1, tt2, tt3], lg_guess, [rad1, rad2, rad3], plx, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True, nspec = 3)
	else:	
		w, spe, pri_spec, sec_spec, tri_spec, pri_mag, sec_mag, tri_mag = make_composite([tt1, tt2, tt3], lg_guess, [rad1, rad2, rad3], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True, nspec = 3)

	itep = interp1d(w, spe)
	spe = itep(wl*1e4)

	pri_ratio =  (np.median(spec)/np.median(spe))
	sec_ratio = (np.median(spec)/np.median(spe)) 

	i1 = interp1d(w, pri_spec)
	pri_spec = i1(wl*1e4)
	pri_spec *= pri_ratio
	i2 = interp1d(w, sec_spec)
	sec_spec = i2(wl*1e4)
	sec_spec *= sec_ratio
	i3 = interp1d(w, tri_spec)
	tri_spec = i2(wl*1e4)
	tri_spec *= sec_ratio


	spe *= np.median(spec)/np.median(spe)

	with open(run + '/params.txt', 'w') as f:
		if len(a) == 8 and dist_fit == True:
			f.write('teff: {} +/- {} + {} +/- {} + {} +/- {}\nradius: {} +/- {} + {} +/- {} + {} +/- {}\nextinction: {}\nparallax: {}\nprimary Kep mag:{}\nsecondary Kep mag:{}\ntertiary Kep mag:{}'.\
				format(tt1, sigma_t1, tt2, sigma_t2, tt3, sigma_t3, rad1, sigma_r1, rad2, sigma_r2, rad3, sigma_r3, te, plx, pri_mag, sec_mag, tri_mag))
		else:
			f.write('teff: {} +/- {} + {} +/- {}\nradius: {} +/- {}\nextinction: {}\nprimary Kep mag:{}\nsecondary Kep mag:{}'.format(tt1, sigma_t1, tt2, sigma_t2, rad2, sigma_r2, te, pri_mag, sec_mag))
	#make a nice figure with the different spectra
	fig, ax = plt.subplots(figsize = (7,6))

	if any(v != 0 for v in real_val):
		ax.plot(wl*1e4, spec, linewidth = 1, label = 'Data: {:.0f}+{:.0f}+{:.0f}K; {:.1f}+{:.1f}+{:.1f}dex'.format(real_val[0], real_val[1], real_val[2], real_val[3], real_val[4], real_val[5]), color = 'k', zorder = 4)
	else:
		ax.plot(wl*1e4, spec, linewidth = 1, label = 'Data', color = 'k', zorder = 4)
	ax.plot(wl*1e4, spe, linewidth = 1, label = 'Composite spectrum', color = 'seagreen', zorder=3.5)
	ax.plot(wl*1e4, pri_spec, linewidth = 1, label = 'Primary: {:.0f}K'.format(tt1), color = 'darkblue', zorder = 3)
	ax.plot(wl*1e4, sec_spec, linewidth = 1, label = 'Secondary: {:.0f}K'.format(tt2), color = 'darkorange', zorder = 3)
	ax.plot(wl*1e4, tri_spec, linewidth = 1, label = 'Tertiary: {:.0f}K '.format(tt3), color = 'crimson', zorder = 3)

	for n in range(len(random_sample)):
		if len(a) == 8 and dist_fit == True:
			t1, t2, t3, e, r1, r2, r3, pl = random_sample[n]
			dist = 1/pl
			log_g_guess = [get_logg(t, matrix) for t in [t1, t2, t3]]
			ww, sspe, ppri_spec, ssec_spec, ttri_spec, ppri_mag, ssec_mag, ttri_mag = make_composite([t1, t2, t3], log_g_guess, [r1, r2, r3], pl, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True, nspec = 3)
		elif len(a) == 8 and dist_fit == False:
			t1, t2, t3, e, r1, ratio1, ratio2, pl = random_sample[n]
			log_g_guess = [get_logg(t, matrix) for t in [t1, t2, t3]]
			ww, sspe, ppri_spec, ssec_spec, ttri_spec, ppri_mag, ssec_mag, ttri_mag = make_composite([t1, t2, t3], log_g_guess, [ratio1, ratio2], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True, nspec = 3)

		else:
			print('spectrum generation error!')
		ite = interp1d(ww, sspe)
		sspe = ite(wl*1e4)

		it1, it2, it3 = interp1d(ww, ppri_spec), interp1d(ww, ssec_spec), interp1d(ww, ttri_spec)
		ppri_spec, ssec_spec, ttri_spec = it1(wl*1e4), it2(wl*1e4), it3(wl*1e4)

		if len(a) == 8:
			ratio1 = rad2
			ratio2 = rad3

		ppri_spec *= pri_ratio; ssec_spec *= sec_ratio; ttri_spec *= sec_ratio

		sspe *= np.median(spec)/np.median(sspe)

		ax.plot(wl*1e4, sspe, linewidth = 0.75, color = 'limegreen', alpha = 0.5, zorder = 2.5, rasterized = True)
		ax.plot(wl*1e4, ppri_spec, linewidth = 0.75, color = 'skyblue', alpha = 0.5, zorder = 2, rasterized = True)
		ax.plot(wl*1e4, ssec_spec, linewidth = 0.75, color = 'gold', alpha = 0.5, zorder = 2, rasterized = True)
		ax.plot(wl*1e4, ttri_spec, linewidth = 0.75, color = 'crimson', alpha = 0.5, zorder = 2, rasterized = True)

	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = 14, direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	# ax.set_xlim(6900, 7500)
	ax.set_xlabel(r'Wavelength ($\AA$)', fontsize = 14)
	ax.set_ylabel('{}'.format(r'Normalized Flux'), fontsize = 14)
	ax.legend(loc = 'best', fontsize = 12)
	plt.tight_layout()
	plt.savefig(run + '/plots/{}_all_spec.pdf'.format(fname))

	############
	#CREATE THE COMPOSITE + DATA PLOT 
	###########

	if not real_val[0] == 0:
		if len(real_val) == 8 and dist_fit == True:
			rt1, rt2, rt3, rr1, rr2, rr3, rpl = real_val
			real_logg = [get_logg(t, matrix) for t in [rt1, rt2, rt3]]
			real_wl, rspec, rpspec, rsspec, rtspec, pri_mag, sec_mag, tri_mag = make_composite([rt1, rt2, rt3], real_logg, [rr1, rr2, rr3], rpl, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True, nspec = 3)
		elif len(real_val) == 8 and dist_fit == False:
			rt1, rt2, rt3, rr1, ratio, ratio2, rpl = real_val
			real_logg = [get_logg(t, matrix) for t in [rt1, rt2, rt3]]
			real_wl, rspec, rpspec, rsspec, rtspec, pri_mag, sec_mag, tri_mag = make_composite([rt1, rt2, rt3], real_logg, [ratio, ratio2], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True, nspec = 3)
		else:
			print('spec generation error')
		rspec *= np.median(spec)/np.median(rspec[np.where((real_wl < max(wl * 1e4)) & (real_wl > min(wl * 1e4)))])

	fig, ax = plt.subplots()
	ax.plot(wl*1e4, spec, linewidth = 1, label = 'Data spectrum', color = 'navy', zorder = 0)
	ax.plot(wl*1e4, spe, linewidth = 1, label = 'Model: {:.0f}K + {:.0f}K'.format(tt1, tt2), color = 'xkcd:sky blue', zorder=1)
	if not real_val[0] == 0:
		ax.plot(real_wl, rspec, linewidth = 1, color = 'xkcd:grass green', label = 'B15 values: {:.0f}K + {:.0f}K; {:.1f}dex + {:.1f}dex'.format(rt1, rt2, rl1, rl2))
	ax.set_xlim(8500, 8700)
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	plt.xlabel(r'Wavelength (\AA)', fontsize = 13)
	plt.ylabel('Normalized flux', fontsize = 13)
	plt.legend(loc = 'best', fontsize = 13)
	plt.tight_layout()
	plt.savefig(run + '/plots/bestfit_spec_post_mcmc.pdf')
	plt.close()


	############
	#COMPUTE AND CREATE THE KEPLER CONTRAST AND CORRECTION FACTOR PLOTS
	###########

	kep_sample = sample[np.random.choice(len(sample), size = int(1500), replace = False), :]
	kep_sample = kep_sample[np.where((kep_sample[:,2] > 2900) & (kep_sample[:,1] > 2900) & (kep_sample[:,2] < 6200) & (kep_sample[:,1] < 6200) & (kep_sample[:,0] < 6200))]

	kep_contrast = []; kep_contrast2 = []; kep_rad = []; kep_rad2 = []
	for n in range(len(kep_sample)):
		if len(a) == 8 and dist_fit == True:
			tt1, tt2, tt3, ex, rad1, rad2, rad3, plx = kep_sample[n,:]
			logg_sample = [get_logg(t, matrix) for t in [tt1, tt2, tt3]]
			w, spe, pri_spec, sec_spec, tri_spec, pri_mag, sec_mag, tri_mag = make_composite([tt1, tt2, tt3], logg_sample, [rad1, rad2, rad3], plx, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True, nspec = 3)
		elif len(a) == 8 and dist_fit == False:
			tt1, tt2, tt3, ex, rad1, rad2, rad3, plx = kep_sample[n,:]
			logg_sample = [get_logg(t, matrix) for t in [tt1, tt2, tt3]]
			w, spe, pri_spec, sec_spec, tri_spec, pri_mag, sec_mag, tri_spec = make_composite([tt1, tt2, tt3], logg_sample, [rad2, rad3], False, fr[2], fr[5], [min(wl), max(wl)], sp, ctm, ptm, tmi, tma, vs, models = models, plot = True)
		else:	
			print('kep magnitude error')

		kep_rad.append(rad2); kep_rad2.append(rad3)
		kep_contrast.append(sec_mag[0]-pri_mag[0])
		kep_contrast2.append(tri_mag[0] - pri_mag[0])

	kep_contrast = np.array(kep_contrast); kep_rad = np.array(kep_rad); kep_contrast2 = np.array(kep_contrast2); kep_rad2 = np.array(kep_rad2)

	nbins = 110
	contrast_bins = np.linspace(min(kep_contrast), max(kep_contrast), nbins)
	contrast_count = np.zeros(len(contrast_bins))

	for t in kep_contrast:
		for b in range(len(contrast_bins) - 1):
			if contrast_bins[b] <= t < contrast_bins[b + 1]:
				contrast_count[b] += 1

	contrast_bins2 = np.linspace(min(kep_contrast2), max(kep_contrast2), nbins)
	contrast_count2 = np.zeros(len(contrast_bins2))


	for t in kep_contrast2:
		for b in range(len(contrast_bins2) - 1):
			if contrast_bins2[b] <= t < contrast_bins2[b + 1]:
				contrast_count2[b] += 1


	kep_mean = np.abs(np.percentile(kep_contrast, 50)); kep_84 = np.abs(np.percentile(kep_contrast, 84)); kep_16 = np.abs(np.percentile(kep_contrast,16))

	fig, ax = plt.subplots(figsize = (4, 4))
	ax.hist(kep_contrast, histtype = 'step', linewidth = 2, color = 'k')
	ax.axvline(kep_84, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(kep_16, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(kep_mean, linestyle = '-', color = 'k', linewidth = 2)
	ax.set_title(r'$\Delta$Kep = {:.3f}$^{{+ {:.3f}}}_{{- {:.3f}}}$'.format(kep_mean, kep_84-kep_mean, kep_mean - kep_16))
	ax.set_xlabel(r'$\Delta$Kep (mag)')
	plt.tight_layout()
	plt.savefig('{}/plots/{}_delta_kep_sec.pdf'.format(run, fname))
	plt.close()

	np.savetxt('{}/kep_contrast.txt'.format(run), kep_contrast)

	kep_mean2 = np.abs(np.percentile(kep_contrast2, 50)); kep2_84 = np.abs(np.percentile(kep_contrast2, 84)); kep2_16 = np.abs(np.percentile(kep_contrast2,16))

	fig, ax = plt.subplots(figsize = (4, 4))
	ax.hist(kep_contrast2, histtype = 'step', linewidth = 2, color = 'k')
	ax.axvline(kep2_84, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(kep2_16, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(kep_mean2, linestyle = '-', color = 'k', linewidth = 2)
	ax.set_title(r'$\Delta$Kep = {:.3f}$^{{+ {:.3f}}}_{{- {:.3f}}}$'.format(kep_mean2, kep2_84-kep_mean2, kep_mean2 - kep2_16))
	ax.set_xlabel(r'$\Delta$Kep (mag)')
	plt.tight_layout()
	plt.savefig('{}/plots/{}_delta_kep_tri.pdf'.format(run, fname))
	plt.close()
	np.savetxt('{}/kep_contrast_tri.txt'.format(run), kep_contrast)

	pri_corr = np.sqrt(1 + 10**(-0.4 * kep_contrast)) #from ciardi+2015; Furlan+2017
	sec_corr = kep_rad * np.sqrt(1 + 10**(0.4 * kep_contrast))
	tri_corr = kep_rad2 * np.sqrt(1 + 10**(0.4*kep_contrast2))

	pc_bins = np.linspace(min(pri_corr), max(pri_corr), nbins)
	sc_bins = np.linspace(min(sec_corr), max(sec_corr), nbins)
	tc_bins = np.linspace(min(tri_corr), max(tri_corr), nbins)

	pc_count = np.zeros(len(pc_bins))
	sc_count = np.zeros(len(sc_bins))
	tc_count = np.zeros(len(tc_bins))

	for t in pri_corr:
		for b in range(len(pc_bins) - 1):
			if pc_bins[b] <= t < pc_bins[b + 1]:
				pc_count[b] += 1

	for t in sec_corr:
		for b in range(len(sc_bins) - 1):
			if sc_bins[b] <= t < sc_bins[b + 1]:
				sc_count[b] += 1

	for t in tri_corr:
		for b in range(len(tc_bins) - 1):
			if tc_bins[b] <= t < tc_bins[b + 1]:
				tc_count[b] += 1

	pri_mean = np.abs(np.percentile(pri_corr, 50)); pri_84 = np.abs(np.percentile(pri_corr, 84)); pri_16 = np.abs(np.percentile(pri_corr,16))

	fig, ax = plt.subplots(figsize = (4, 4))
	ax.hist(pri_corr, histtype = 'step', linewidth = 2, color = 'k')
	ax.axvline(pri_84, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(pri_16, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(pri_mean, linestyle = '-', color = 'k', linewidth = 2)
	ax.set_title(r'$f_{{p, corr}}$ = {:.4f}$^{{+ {:.4f}}}_{{- {:.4f}}}$'.format(pri_mean, pri_84-pri_mean, pri_mean - pri_16))
	ax.set_xlabel(r'Corr. factor (primary)')
	plt.tight_layout()
	plt.savefig(run + '/plots/{}_pri_corr.pdf'.format(fname))
	plt.close()

	np.savetxt(run + '/pri_corr.txt', pri_corr)

	sec_mean = np.abs(np.percentile(sec_corr, 50)); sec_84 = np.abs(np.percentile(sec_corr, 84)); sec_16 = np.abs(np.percentile(sec_corr,16))

	fig, ax = plt.subplots(figsize = (4, 4))
	ax.hist(sec_corr, histtype = 'step', linewidth = 2, color = 'k')
	ax.axvline(sec_84, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(sec_16, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(sec_mean, linestyle = '-', color = 'k', linewidth = 2)
	ax.set_title(r'$f_{{s, corr}}$ = {:.3f}$^{{+ {:.3f}}}_{{- {:.3f}}}$'.format(sec_mean, sec_84-sec_mean, sec_mean - sec_16))
	ax.set_xlabel(r'Corr. factor (secondary)')
	plt.tight_layout()
	plt.savefig(run + '/plots/{}_sec_corr.pdf'.format(fname))
	plt.close()

	np.savetxt(run + '/sec_corr.txt', sec_corr)

	tri_mean = np.abs(np.percentile(tri_corr, 50)); tri_84 = np.abs(np.percentile(tri_corr, 84)); tri_16 = np.abs(np.percentile(tri_corr,16))

	fig, ax = plt.subplots(figsize = (4, 4))
	ax.hist(tri_corr, histtype = 'step', linewidth = 2, color = 'k')
	ax.axvline(tri_84, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(tri_16, linestyle = '--', color = 'k', linewidth = 2)
	ax.axvline(tri_mean, linestyle = '-', color = 'k', linewidth = 2)
	ax.set_title(r'$f_{{t, corr}}$ = {:.3f}$^{{+ {:.3f}}}_{{- {:.3f}}}$'.format(tri_mean, tri_84-tri_mean, tri_mean - tri_16))
	ax.set_xlabel(r'Corr. factor (tertiary)')
	plt.tight_layout()
	plt.savefig(run + '/plots/{}_tri_corr.pdf'.format(fname))
	plt.close()

	np.savetxt(run + '/tri_corr.txt', tri_corr)

	############
	#CREATE ISOCHRONE + COMPONENT PLOT
	###########

	#make this in H-K vs. K space so I can use contrasts to get a mag for each component
	matrix = np.genfromtxt('mist_2mass_old.cmd', autostrip = True)
	#get the age and convert it into Gyr from log(years)
	aage = matrix[:, 1]

	teff5, lum5 = matrix[:,4][np.where(np.array(aage) == 9.0000000000000000)], matrix[:,6][np.where(np.array(aage) == 9.0000000000000000)]
	ma5 = matrix[:,3][np.where(np.array(aage) == 9.0000000000000000)]

	aage = np.array([(10**a)/1e9 for a in aage])

	#get the mass, log(luminosity), effective temp, and log(g)
	ma = matrix[:,3][np.where((aage > 0.1) & (aage < 8))]
	teff = matrix[:, 4][np.where((aage > 0.1) & (aage < 8))]
	lum = matrix[:, 6][np.where((aage > 0.1) & (aage < 8))]
	hmag = matrix[:, 15][np.where((aage > 0.1) & (aage < 8))]
	kmag = matrix[:, 16][np.where((aage > 0.1) & (aage < 8))]
	hk_color = np.array(hmag) - np.array(kmag)
	aage = aage[np.where((aage > 0.1) & (aage < 8))]

	lum, lum5 = [10**l for l in lum], [10**l for l in lum5]; 
	teff, teff5 = [10**t for t in teff], [10**t for t in teff5]

	#remove redundant ages from the age vector
	a1 = [aage[0]]
	for n in range(len(aage)):
		if aage[n] != a1[-1]:
			a1.append(aage[n])

	# #plot the age = 0 temperature vs. luminosity 
	# fig, ax = plt.subplots()

	# #now for all the other ages fill an array with the single valued age, get the temperature and convert it from log
	# #then plot it versus the correct luminosity
	# #tagging each one with the age and color coding it 
	# for n in np.arange(0, len(a1), 4):
	# 	a2 = np.full(len(np.where(aage == a1[n])[0]), a1[n])

	# 	if n == 0:
	# 		ax.plot(np.array(teff)[np.where(np.array(aage) == a1[n])], np.log10(np.array(lum)[np.where(np.array(aage) == a1[n])]), color = cm.plasma(a1[n]/10), zorder = 0, label = 'MS')#, label = '{}'.format(int(np.around(a1[n]))))
	# 	else:
	# 		ax.plot(np.array(teff)[np.where(np.array(aage) == a1[n])], np.log10(np.array(lum)[np.where(np.array(aage) == a1[n])]), color = cm.plasma(a1[n]/10), zorder = 0)#, label = '{}'.format(int(np.around(a1[n]))))

	# # print('luminosity interpolation', tt1, min(teff5[:200]), max(teff5[:200]))
	l_intep = interp1d(teff5[:200], lum5[:200])
	# pri_lum = l_intep(tt1)

	# sigma_sb = 5.670374e-5 #erg/s/cm^2/K^4
	# lsun = 3.839e33 #erg/s 
	# rsun = 6.955e10
	# pri_rad = np.sqrt(pri_lum*lsun/(4 * np.pi * sigma_sb * tt1**4)) #cm

	# sec_rad = ratio1 * pri_rad
	# sec_lum = (4 * np.pi * sec_rad**2 * sigma_sb * tt2**4)/lsun #solar luminosity

	# tri_rad = ratio2 * pri_rad
	# tri_lum = (4 * np.pi * tri_rad**2 * sigma_sb * tt3**4)

	# ax.scatter(tt1, np.log10(pri_lum), marker = 'x', color = 'darkgray', s = 60, label = 'Primary')
	# ax.scatter(tt2, np.log10(sec_lum), marker = 'x', color = 'darkorange', s = 50, label = 'Secondary')
	# ax.scatter(tt3, np.log10(tri_lum), marker = 'x', color = 'crimson', s = 50, label = 'Tertiary')
	# ax.set_xlabel(r'T$_{eff}$ (K)', fontsize = 16)
	# ax.set_ylabel(r'$\log_{10}$(L (L$_{\odot}$))', fontsize = 16)
	# # ax.set_yscale('log')
	# ax.set_xlim(max(tt1, tt2, tt3) + 200, min(tt2, tt3, tt1) - 200)
	# ax.set_ylim(np.log10(min(pri_lum, sec_lum, tri_lum) - 0.1), np.log10(max(pri_lum, sec_lum, tri_lum) + 1))
	# plt.minorticks_on()
	# ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	# ax.tick_params(bottom=True, top =True, left=True, right=True)
	# ax.tick_params(which='both', labelsize = 16, direction='in')
	# ax.tick_params('both', length=8, width=1.5, which='major')
	# ax.tick_params('both', length=4, width=1, which='minor')
	# ax.legend(loc = 'best', fontsize = 13)
	# # plt.gca().invert_yaxis()
	# fig.tight_layout()
	# #save the figure in the run directory
	# plt.savefig(run + '/plots/{}_isochrone.pdf'.format(fname))
	# plt.close()


	ma_intep = interp1d(teff5[:200], ma5[:200])

	pmass_posterior = []; plum_posterior = []
	smass_posterior = []; slum_posterior = []
	tmass_posterior = []; tlum_posterior = []

	for n in range(len(kep_sample)):
		tt1, tt2, tt3, ex, rad1, rad2, rad3, plx = kep_sample[n,:]
		interp_map = ma_intep(tt1)
		interp_mas = ma_intep(tt2)
		interp_mat = ma_intep(tt3)
		pmass_posterior.append(float(interp_map)); smass_posterior.append(float(interp_mas)); tmass_posterior.append(float(interp_mat))

		interp_plum = l_intep(tt1)
		interp_slum = l_intep(tt2)
		interp_tlum = l_intep(tt3)
		plum_posterior.append(float(interp_plum)); slum_posterior.append(float(interp_slum)); tlum_posterior.append(float(interp_tlum))

	savefig_titles = ['primary_mass_posterior', 'secondary_mass_posterior', 'tertiary_mass_posterior', 'primary_lum_posterior', 'secondary_lum_posterior', 'tertiary_lum_posterior']
	figure_titles = ['M$_{{pri}}$', 'M$_{{sec}}$',  'M$_{{tri}}$', 'L$_{{pri}}$', 'L$_{{sec}}$', 'L$_{{tri}}$']
	xax_label = ['M$_{{pri}} (M_{{\odot}})$', 'M$_{{sec}} (M_{{\odot}})$', 'M$_{{tri}} (M_{{\odot}})$', 'L$_{{pri}} (L_{{\odot}})$', 'L$_{{sec}} (L_{{\odot}})$', 'L$_{{tri}} (L_{{\odot}})$']
	for n, posterior in enumerate([pmass_posterior, smass_posterior, tmass_posterior, plum_posterior, slum_posterior, tlum_posterior]):

		sc_bins = np.linspace(min(posterior), max(posterior), nbins)
		sc_count = np.zeros(len(sc_bins))

		for t in posterior:
			for b in range(len(sc_bins) - 1):
				if sc_bins[b] <= t < sc_bins[b + 1]:
					sc_count[b] += 1
		sec_mean = np.abs(np.percentile(np.array(posterior), 50))
		sec_84 = np.abs(np.percentile(np.array(posterior), 84))
		sec_16 = np.abs(np.percentile(np.array(posterior),16))

		fig, ax = plt.subplots(figsize = (4, 4))
		ax.hist(posterior, histtype = 'step', linewidth = 2, color = 'k')
		ax.axvline(sec_84, linestyle = '--', color = 'k', linewidth = 2)
		ax.axvline(sec_16, linestyle = '--', color = 'k', linewidth = 2)
		ax.axvline(sec_mean, linestyle = '-', color = 'k', linewidth = 2)
		ax.set_title(r'{} = {:.3f}$^{{+ {:.3f}}}_{{- {:.3f}}}$'.format(figure_titles[n], sec_mean, sec_84-sec_mean, sec_mean - sec_16))
		ax.set_xlabel(r'{}'.format(xax_label[n]))
		plt.tight_layout()
		plt.savefig(run + '/plots/{}.pdf'.format(savefig_titles[n]))
		plt.close()

		np.savetxt(run + '/' + savefig_titles[n] + '.txt', np.array(posterior))


	return

def main(argv):
	argument_list = argv[1:]
	short_options = 'f:o:e:' #filename, optimize y/n, emcee y/n
	long_options = 'file =, optimize =, emcee ='
	arguments, values = getopt.getopt(argument_list, short_options, long_options)

	parkey, parfile = arguments[0]

	pardict = {}
	with open(parfile) as fi:
		for line in fi:
			if not line.startswith('#') and not line.strip() == '':
				# print(line.split(' ')[0:2])
				(key, val) = line.split(' ')[0:2]
				val = val.split('\t')[0]
				# print(key, val)
				pardict[str(key)] = val

	# print(pardict)
	models = pardict['models']; res = int(pardict['res'])
	try:
		mask = pardict['mask']
	except:
		mask = 'f'

	try:
		rp = pardict['rad_prior']
	except:
		rp = 'f'

	if 't' in rp.lower():
		rp = True
	else:
		rp = False

	# vs = Table(fits.getdata('Data/vegaspec.fits'))
	vs2 = synphot.spectrum.SourceSpectrum.from_file('Data/vegaspec.fits')
	matrix = np.genfromtxt('mist_2mass_old.cmd', autostrip = True)

	matrix[:,4] = [10**t for t in matrix[:,4]]
	matrix[:,6] = [10**l for l in matrix[:,6]]

	data_wl, dsp, de = np.genfromtxt(pardict['filename'], unpack = True)

	# data_wl /= 1e4

	if 't' in mask.lower():
		dsp = np.concatenate((dsp[np.where(data_wl <= 0.6860)], dsp[np.where((data_wl >= 0.6880) & (data_wl <= 0.7600))], dsp[np.where((data_wl >= 0.7660) & (data_wl <= 0.8210))], dsp[np.where(data_wl > 0.8240)]))
		de = np.concatenate((de[np.where(data_wl <= 0.6860)], de[np.where((data_wl >= 0.6880) & (data_wl <= 0.7600))], de[np.where((data_wl >= 0.7660) & (data_wl <= 0.8210))], de[np.where(data_wl > 0.8240)]))
		data_wl = np.concatenate((data_wl[np.where(data_wl <= 0.6860)], data_wl[np.where((data_wl >= 0.6880) & (data_wl <= 0.7600))], data_wl[np.where((data_wl >= 0.7660) & (data_wl <= 0.8210))], data_wl[np.where(data_wl > 0.8240)]))


	data_wl, dsp, de = data_wl[np.where((data_wl > float(pardict['spmin'])) & (data_wl < float(pardict['spmax'])))], \
		dsp[np.where((data_wl > float(pardict['spmin'])) & (data_wl < float(pardict['spmax'])))], \
			de[np.where((data_wl > float(pardict['spmin'])) & (data_wl < float(pardict['spmax'])))]

	de /= np.median(dsp)
	dsp /= np.median(dsp)

	data = [data_wl, dsp]

	t1 = time.time()
	specs = spec_interpolator([float(pardict['spmin'])*1e4, float(pardict['spmax'])*1e4], [int(pardict['tmin']), int(pardict['tmax'])], [4,5.5], \
		[int(pardict['specmin']), int(pardict['specmax'])], resolution = res, models = models)
	print('time to read in specs:', time.time() - t1)

	plx, plx_err, dist_true = float(pardict['plx']), float(pardict['plx_err']), pardict['dist_fit']
	if 't' in dist_true.lower():
		dist_true = True
	else:
		dist_true = False


	mags = list([float(p) for p in pardict['cmag'].strip('[]').split(',')])
	me = list([float(p) for p in pardict['cerr'].strip('[]').split(',')])
	filts = np.array([p.strip('\\') for p in pardict['cfilt'].strip('[] ').split('\'')])
	filts = np.array([p for p in filts if len(p) >= 1 and not p == ','])
	try:
		oldphot = list([float(p) for p in pardict['pmag'].strip('[]').split(',')])
	except:
		if 'np.nan' in pardict['pmag']:
			oldphot = list([p for p in pardict['pmag'].strip('[]').split(',')])
			for n,p in enumerate(oldphot):
				if p == 'np.nan':
					oldphot[n] = np.nan
				else:
					oldphot[n] = float(p)

					
	phot_err = list([float(p) for p in pardict['perr'].strip('[]').split(',')])
	phot_filt = np.array([p.strip('\\') for p in pardict['pfilt'].strip('[] ').split('\'')])
	phot_filt = np.array([p for p in phot_filt if len(p) >= 1 and not p == ','])
	phot = np.zeros(len(oldphot))


	#Now that I'm calculating photometry manually I don't think I need to do this, since the ZPs should be in ab mag
	ab_to_vega = {'u':0.91,'g':-0.08,'r':0.16,'i':0.37,'z':0.54} #m_AB-m_vega from Blanton et al 2007
	kic_to_sdss_slope = {'g':0.0921, 'r':0.0548, 'i':0.0696, 'z':0.1587}
	kic_to_sdss_int = {'g':-0.0985,'r':-0.0383,'i':-0.0583,'z':-0.0597}
	kic_to_sdss_color = {'g':'g-r', 'r':'r-i','i':'r-i','z':'i-z'}

	# #if we have mAB, m_vega = m_ab - ab_to_vega
	#now everything will be in vegamag
	if not 'synth' in parfile:
		for n, p in enumerate(phot_filt):
			if 'sdss' in p.lower():
				color = oldphot[np.where('sdss,' + kic_to_sdss_color[p.split(',')[1]].split('-')[0] == phot_filt)[0][0]] - oldphot[np.where('sdss,' + kic_to_sdss_color[p.split(',')[1]].split('-')[1] == phot_filt)[0][0]]
				phot[n] = kic_to_sdss_int[p.split(',')[1]] + kic_to_sdss_slope[p.split(',')[1]]*color + oldphot[n]
				phot[n] = phot[n] #- ab_to_vega[p.split(',')[1]]
			else:
				phot[n] = oldphot[n]
	else:
		phot = np.array(oldphot) #- np.array([-0.08,0.16,0.37,0.54,0,0,0]) #-(5 * np.log10((1/plx)/10) - 5)

	#1" parallax = 1/D (in pc)
	av = float(pardict['av']); av_err = float(pardict['av_err'])

	#read in the ra and dec in deg from epoch 2015.5 (thanks, exoFOP)
	ra, dec = float(pardict['ra']), float(pardict['dec'])

	nwalk1, cutoff, nburn, nstep = int(pardict['nwalk']), 1, 1, int(pardict['nstep'])

	nspec, ndust = int(pardict['nspec']), int(pardict['ndust'])
	
	#give everything in fluxes
	fr = [mags, me, filts, phot, phot_err, phot_filt]

	tmi, tma = np.inf, 0

	wls, tras, n_res_el, cwl = [], [], [], []
	for f in fr[2]:
		w, t, re, c = get_transmission(f, res)
		wls.append(list(w)); tras.append(list(t)); n_res_el.append(re); cwl.append(c)
		if min(w) < tmi:
			tmi = min(w)
		if max(w) > tma:
			tma = max(w)

	phot_wls, phot_tras, phot_resel, phot_cwl = [], [], [], []
	for p in fr[5]:
		w, t, re, c = get_transmission(p, res)
		phot_wls.append(list(w)); phot_tras.append(list(t)); phot_resel.append(re); phot_cwl.append(c)
		if min(w) < tmi:
			tmi = min(w)
		if max(w) > tma:
			tma = max(w)

	ctm, ptm = [wls, tras, n_res_el, cwl], [phot_wls, phot_tras, phot_resel, phot_cwl]

	# #uncomment this code and run param_make_synth.txt to make the synthetic composite data sets
	# for n, t2 in enumerate([3025,3225,3425,3625,3800]):#[3225, 3425, 3625, 3825, 4025, 4175]):#
	# 	t1 =  3850  # #4500  . 4200
	# 	r1 =  0.4994   # 0.6162 
	# 	r2 = [0.1546, 0.2149, 0.3048, 0.3910, 0.4745][n] #[0.2149, 0.3048, 0.3910, 0.4870, 0.5791, 0.6039][n] #
	# 	lg1 = 4.76 # 4.67 
	# 	lg2 = [5.16, 5.06, 4.96, 4.87, 4.79][n] #[5.06, 4.96, 4.87, 4.77, 4.70, 4.68][n] #

	# for n, t2 in enumerate([3001, 3501, 4001, 4501]):
	# 	t1 =  5000
	# 	r1 =  0.755  
	# 	r2 = [0.149, 0.331, 0.569, 0.666][n] 
	# 	lg1 = 4.59
	# 	lg2 = [5.16, 4.94, 4.7, 4.64][n] 

	# for n, t2 in enumerate([3501, 4001, 4501, 5001, 5501]):
	# 	t1 =  6000
	# 	r1 =  1.043 
	# 	r2 = [0.331, 0.569, 0.666, 0.755, 0.851][n]
	# 	lg1 = 4.43 
	# 	lg2 = [4.94, 4.7, 4.64, 4.59, 4.54][n] 

	# for n, t1 in enumerate([3800, 4800, 5800]):
	# 	r1 = [0.5, 0.75, 1][n]

	# 	for k, t2 in enumerate([3225, 3625, 4001, 4501, 5001, 5501]):
	# 		r2 = [0.2149, 0.3910, 0.569, 0.666, 0.755, 0.851][k]

	# 		for p, t3 in enumerate([3025, 3425, 3800, 4501, 5001]):
	# 			r3 = [0.1546, 0.3048, 0.4745, 0.569, 0.666][p]

	# 			if t2 < t1 and t3 < t2:
	# 				lg_guess = np.array([get_logg(t, matrix) for t in [t1, t2, t3]])
	# 				comp_wl, comp_spec, contrast, phot_cwl, photometry = make_composite([t1, t2, t3], lg_guess, [r1,r2/r1, r3/r1], plx, ['880','Kp','880','Kp'], ['sdsss,r','sdss,i','sdss,z','J', 'H', 'K'], [min(data_wl),max(data_wl)], specs, ctm, ptm, tmi, tma, vs2, nspec = nspec)  
	# 				print(contrast, photometry, t1, t2, t3)
	# 				# print(min(comp_wl), min(data_wl), max(comp_wl), max(data_wl))
	# 				# # plt.plot(comp_wl, comp_spec/np.median(comp_spec[np.where((comp_wl < max(data_wl*1e4)) & (comp_wl > min(data_wl*1e4)))]), linewidth = 1)
	# 				# # plt.plot(data_wl*1e4, dsp, linewidth = 1)
	# 				# # plt.xlim(5500,9000)
	# 				# # plt.savefig('test_data.pdf')
	# 				itep = interp1d(comp_wl, comp_spec); comp_spec = itep(data_wl*1e4)
	# 				err = np.random.normal(0, 0.01*comp_spec)

	# 				np.savetxt('Data/synth_spec_{}_{}_{}.txt'.format(t1, t2, t3), np.column_stack((data_wl, comp_spec + err, err)))

	dirname, fname = pardict['dirname'], pardict['fname']

	try:
		os.mkdir(dirname)
	except:
		pass;
	try:
		os.mkdir(dirname + '/plots')
	except:
		pass;

	optkey, optval = arguments[1]
	if optval == 'True':
		optimize_fit(dirname, data, de, specs, nwalk1, fr, [plx, plx_err], [av, av_err], res, ctm, ptm, tmi, tma, vs2, matrix, ra, dec, nspec = nspec, cutoff = cutoff, nstep = nstep, nburn = nburn, con = False, models = models, dist_fit = dist_true, rad_prior = rp)
		
		if nspec == 2:
			plot_fit(dirname, data, specs, fr, ctm, ptm, tmi, tma, vs2, matrix, models = models, dist_fit = dist_true)
		elif nspec == 3:
			plot_fit3(dirname, data, specs, fr, ctm, ptm, tmi, tma, vs2, matrix, models = models, dist_fit = dist_true)

		print('optimization complete')

	emceekey, emceeval = arguments[2]
	if emceeval == 'True':
		chisqs, pars = np.genfromtxt(dirname + '/optimize_cs.txt'), np.genfromtxt(dirname + '/optimize_res.txt')

		cs_idx = sorted(chisqs)[:int(len(chisqs)*1/3)]

		idx = np.array([int(np.where(chisqs == c)[0]) for c in cs_idx])

		p0 = np.array(pars[idx])

		if nspec == 3:
			p0 += np.random.normal(np.zeros(np.shape(p0)), 0.05*np.abs(p0), np.shape(p0))

		nwalkers, nsteps, ndim, nburn = len(p0), int(pardict['nsteps']), len(pars[0]), int(pardict['nburn'])

		real_val = list([float(p) for p in pardict['real_values'].strip('[]\n').split(',')])

		if nspec == 2:
			title_format = ['.0f', '.0f', '.2f', '.2f', '.2f', '.2f', '.2f']
		elif nspec == 3:
			title_format = ['.0f', '.0f', '.0f', '.2f', '.2f', '.2f', '.2f', '.2f']

		a = run_emcee(dirname, fname, nwalkers, nsteps, ndim, nburn, p0, fr, nspec, ndust, data, de, res, [min(data_wl), max(data_wl)], specs, real_val, ctm, ptm, tmi, tma, vs2, title_format, matrix, ra, dec,\
				nthin=100, w = 'aa', du = False, prior = [*np.zeros(len(pars[0])*2-2),plx,plx_err], models = models, av = True, dist_fit = dist_true, rad_prior = rp)
		
		a = np.genfromtxt(dirname + '/samples.txt')
		dw, ds, de = np.genfromtxt(pardict['filename']).T
		dw, ds, de = dw[np.where((dw > float(pardict['spmin'])) & (dw < float(pardict['spmax'])))], \
		ds[np.where((dw > float(pardict['spmin'])) & (dw < float(pardict['spmax'])))], \
		de[np.where((dw > float(pardict['spmin'])) & (dw < float(pardict['spmax'])))]
		data = [dw, ds, de]

		if 't' in mask.lower():
			tell_kw = True
		else:
			tell_kw = False

		if int(nspec) == 2:
			plot_results(fname, a, dirname, data, specs, fr, ctm, ptm, tmi, tma, vs2, real_val, plx, matrix, models = models, dist_fit = dist_true, res = 1700, tell = tell_kw)
		if int(nspec) == 3:
			plot_results3(fname, a, dirname, data, specs, fr, ctm, ptm, tmi, tma, vs2, real_val, plx, matrix, models = models, dist_fit = dist_true, res = 1700, tell = tell_kw)

	return 

if __name__ == "__main__":
	main(sys.argv)
