#Kendall Sullivan

#EMCEE VERSION OF MCMC CODE

#TO DO: Write add_disk function, disk/dust to possible fit params

#20190522: Added extinction with a fixed value to model fitting (prior to fit), updated models to theoretical PHOENIX BT-SETTL models with the 
#CFIST line list downloaded from the Spanish Virtual Observatory "Theoretical Tools" resource. 
#20190901 (ish) Updated model spectra to be the PHOENIX BT-SETTL models with the CFIST 2011 line list downloaded from France Allard's website
#and modified from the FORTRAN format into more modern standard text file format before using
#20200514 Commented everything that's out of beta/active development (or at the very least mostly done) - that goes through the end of get_spec
#the remaining content probably needs to be pared down and likely should be troubleshot fairly carefully

"""
.. module:: model_fit_tools_v2
   :platform: Unix, Mac
   :synopsis: Large package with various spectral synthesis and utility tools.

.. moduleauthor:: Kendall Sullivan <kendallsullivan@utexas.edu>

Dependencies: numpy, pysynphot, matplotlib, astropy, scipy, PyAstronomy, emcee, corner, extinction.
"""

import numpy as np
# import pysynphot as ps
import matplotlib.pyplot as plt
from astropy.io import fits
import os 
from glob import glob
from astropy import units as u
#from matplotlib import rc
from itertools import permutations 
import time, sys
import scipy.stats
from mpi4py import MPI
import timeit
from PyAstronomy import pyasl
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy import ndimage
import emcee
import corner
import extinction
import time
from multiprocessing import Pool
from schwimmbad import MPIPool

def update_progress(progress):
	"""Displays or updates a console progress bar

	Args:
		Progress (float): Accepts a float between 0 and 1. Any int will be converted to a float.

	Note:
		A value under 0 represents a 'halt'.
		A value at 1 or bigger represents 100%

	"""
	barLength = 10 # Modify this to change the length of the progress bar
	status = ""
	if isinstance(progress, int):
		progress = float(progress)
	if not isinstance(progress, float):
		progress = 0
		status = "error: progress var must be float\r\n"
	if progress < 0:
		progress = 0
		status = "Halt...\r\n"
	if progress >= 1:
		progress = 1
		status = "Done...\r\n"
	block = int(round(barLength*progress))
	text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
	sys.stdout.write(text)
	sys.stdout.flush()

def bccorr(wl, bcvel, radvel):
	"""Calculates a barycentric velocity correction given a barycentric and/or a radial velocity (set the unused value to zero)

	Args: 
		wl (list): wavelength vector.
		bcvel (float): a barycentric or heliocentric velocity.
		radvel (float): a systemic radial velocity.

	Note:
		Velocities are in km/s.
		If system RV isn't known, that value can be zero.

	Returns: 
		lam_corr (list): a wavelength vector corrected for barycentric and radial velocities.

	"""
	lam_corr = []
	for w in wl:
		lam_corr.append(w * (1. + (bcvel - radvel)/3e5))
	return lam_corr

def extinct(wl, spec, av, rv = 3.1, unit = 'aa'):
	"""Uses the package "extinction" to calculate an extinction curve for the given A_v and R_v, 
	then converts the extinction curve to a transmission curve
	and uses that to correct the spectrum appropriately.
	Accepted units are angstroms ('aa', default) or microns^-1 ('invum').

	Args:
		wl (list): wavelength array
		spec (list): flux array
		av (float): extinction in magnitudes
		rv (float): Preferred R_V, defaults to 3.1
		unit (string): Unit to use. Accepts angstroms "aa" or inverse microns "invum". Defaults to angstroms.

	Returns:
		spec (list): a corrected spectrum vwith no wavelength vector. 

	"""
	ext_mag = extinction.fm07(wl, av, unit)
	spec = extinction.apply(ext_mag, spec)
	return np.array(spec)
	
def plots(wave, flux, l, lw=1, labels=True, xscale='log', yscale='log', save=False):
	"""makes a basic plot - input a list of wave and flux arrays, and a label array for the legend.
	If you want to label your axes, set labels=True and enter them interactively.
	You can also set xscale and yscale to what you want, and set it to save if you'd like.
	Natively creates a log-log plot with labels but doesn't save it.
	
	Args:
		wave (list): wavelength array
		flux (list): flux array
		l (list): array of string names for legend labels.
		lw (float): linewidths for plot. Default is 1.
		labels (boolean): Toggle axis labels. Initiates interactive labeling. Defaults to True.
		xscale (string): Set x axis scale. Any matplotlib scale argument is allowed. Default is "log".
		yscale (string): Set y axis scale. Any matplotlib scale argument is allowed. Default is "log".
		save (boolean): Saves figure in local directory with an interactively requested title. Defaults to False.
	
	Returns:
		None

	"""
	fig, ax = plt.subplots()
	for n in range(len(wave)):
		ax.plot(wave[n], flux[n], label = l[n], linewidth=lw)
	if labels == True:
		ax.set_xlabel(r'{}'.format(input('xlabel? ')), fontsize=13)
		ax.set_ylabel(r'{}'.format(input('ylabel? ')), fontsize=13)
		ax.set_title(r'{}'.format(input('title? ')), fontsize=15)
	ax.tick_params(which='both', labelsize='larger')
	ax.set_xscale(xscale)
	ax.set_yscale(yscale)
	ax.legend()

	plt.show()
	if save == True:
		plt.savefig('{}.pdf'.format(input('title? ')))

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
	"""Calculates chi square values of a model and data with a given variance.

	Args:
		model (list): model array.
		data (list): data array. Must have same len() as model array.
		variance (float or list): Data variance. Defaults to 10.

	Returns: 
		cs (float): Reduced chi square value.

	"""
	if var == 0:
		var = 10
	#make sure that the two arrays are comparable
	if len(data) == len(model):
		#if there's a variance array, iterate through it as i iterate through the model and data
		if np.size(var) > 1:
			#calculate the chi square vector using xs = (model - data)^2/variance^2 per pixel
			xs = [((model[n] - data[n])**2)/var[n]**2 for n in range(len(model))]
		#otherwise do the same thing but using the same variance value everywhere
		else:
			xs = [((model[n] - data[n])**2)/var**2 for n in range(len(model))]
		#return the chi square vector
		return np.asarray(xs)#np.sum(xs)/len(xs)
	#if the two vectors aren't the same length, yell at me
	else:
		return('data must be equal in length to model')

def shift(wl, spec, rv, bcarr, **kwargs):
	"""for bccorr, use bcarr as well, which should be EITHER:
	1) the pure barycentric velocity calculated elsewhere OR
	2) a dictionary with the following entries (all as floats, except the observatory name code, if using): 
	{'ra': RA (deg), 'dec': dec (deg), 'obs': observatory name or location of observatory, 'date': JD of midpoint of observation}
	The observatory can either be an observatory code as recognized in the PyAstronomy.pyasl.Observatory list of observatories,
	or an array containing longitude, latitude (both in deg) and altitude (in meters), in that order.

	To see a list of observatory codes use "PyAstronomy.pyasl.listobservatories()".
	
	Args:
		wl (list): wavelength array
		spec (list): flux array
		rv (float): Rotational velocity value
		bcarr (list): if len = 1, contains a precomputed barycentric velocity. Otherwise, should 
			be a dictionary with the following properties: either an "obs" keyword and code from pyasl
			or a long, lat, alt set of floats identifying the observatory coordinates.  

	Returns:
		barycentric velocity corrected wavelength vector using bccorr().

	"""
	if len(bcarr) == 1:
		bcvel = bcarr[0]
	if len(bcarr) > 1:
		if isinstance(bcarr['obs'], str):
			try:
				ob = pyasl.observatory(bcarr['obs'])
			except:
				print('This observatory code didn\'t work. Try help(shift) for more information')
			lon, lat, alt = ob['longitude'], ob['latitude'], ob['altitude']
		if np.isarray(bcarr['obs']):
			lon, lat, alt = bcarr['obs'][0], bcarr['obs'][1], bcarr['obs'][2]
		bcvel = pyasl.helcorr(lon, lat, alt, bcarr['ra'], bcarr['dec'], bcarr['date'])[0]

	wl = bccorr()

	return ''

def broaden(even_wl, modelspec_interp, res, vsini = 0, limb = 0, plot = False):
	"""Adds resolution, vsin(i) broadening, taking into account limb darkening.

	Args: 
		even_wl (list): evenly spaced model wavelength vector
		modelspec_interp (list): model spectrum vector
		res (float): desired spectral resolution
		vsini (float): star vsin(i)
		limb (float): the limb darkening coeffecient
		plot (boolean): if True, plots the full input spectrum and the broadened output. Defaults to False.

	Returns:
		a tuple containing an evenly spaced wavelength vector spanning the width of the original wavelength range, and a corresponding flux vector

	"""

	#regrid by finding the smallest wavelength step 
	mindiff = np.inf

	for n in range(1, len(even_wl)):
		if even_wl[n] - even_wl[n-1] < mindiff:
			mindiff = even_wl[n] - even_wl[n-1]

	#interpolate the input values
	it = interp1d(even_wl, modelspec_interp)

	#make a new wavelength array that's evenly spaced with the smallest wavelength spacing in the input wl array
	w = np.arange(min(even_wl), max(even_wl), mindiff)

	sp = it(w)

	#do the instrumental broadening and truncate the ends because they get messy
	broad = pyasl.instrBroadGaussFast(w, sp, res, maxsig=5)
	broad[0:5] = broad[5] 
	broad[len(broad)-10:len(broad)] = broad[len(broad) - 11]

	#if I want to impose stellar parameters of v sin(i) and limb darkening, do that here
	if vsini != 0 and limb != 0:
		rot = pyasl.rotBroad(w, broad, limb, vsini)#, edgeHandling='firstlast')
	#otherwise just move on
	else:
		rot = broad

	#Make a plotting option just in case I want to double check that this is doing what it's supposed to
	if plot == True:

		plt.figure()
		plt.plot(w, sp, label = 'model')
		plt.plot(w, broad, label = 'broadened')
		plt.plot(w, rot, label = 'rotation')
		plt.legend(loc = 'best')
		plt.xlabel('wavelength (angstroms)')
		plt.ylabel('normalized flux')
		plt.savefig('rotation.pdf')

	#return the wavelength array and the broadened flux array
	return w, rot

def redres(wl, spec, factor):
	"""Imposes instrumental resolution limits on a spectrum and wavelength array
	Assumes evenly spaced wl array

	"""
	new_stepsize = (wl[1] - wl[0]) * factor

	wlnew = np.arange(min(wl), max(wl), new_stepsize)

	i = interp1d(wl, spec)
	specnew = i(wlnew)

	#return the reduced spectrum and wl array
	return wlnew, specnew

def rmlines(wl, spec, **kwargs):
	"""Edits an input spectrum to remove emission lines

	Args: 
		wl (list): wavelength
		spec (list): spectrum.
		add_line (boolean): to add more lines to the linelist (interactive)
		buff (float): to change the buffer size, input a float here. otherwise the buffer size defaults to 15 angstroms
		uni (boolean): specifies unit for input spectrum wavelengths (default is microns) [T/F]
		conv (boolean): if unit is true, also specify conversion factor (wl = wl * conv) to microns

	Returns: 
		spectrum with the lines in the linelist file removed if they are in emission.

	"""
	#reads in a linelist, which contains the names, transition, and wavelength of each emission line
	names, transition, wav = np.genfromtxt('linelist.txt', unpack = True, autostrip = True)
	#define the gap to mark out
	space = 1.5e-3 #15 angstroms -> microns

	#check the kwargs
	for key, value in kwargs.items():
		#if i want to add a line, use add_lines, which then lets me append a new line
		if key == add_line:
			wl.append(input('What wavelengths (in microns) do you want to add? '))
		#if i want to change the size of region that's removed, use a new buffer
		if key == buff:
			space = value
		#if i want to change the unit, use uni to do so
		if key == uni:
			wl = wl * value

	diff = wl[10] - wl[9]

	#for each line, walk trhough and remove the line, replacing it with the mean value of the end points of the removed region
	for line in wav:
		end1 = find_nearest(wl, line-space)
		end2 = find_nearest(wl, line+space)
		if wl[end1] > min(wl) and wl[end2] < max(wl) and (end1, end2)> (0, 0) and (end1, end2) < (len(wl), len(wl)):
			for n in range(len(wl)):
				if wl[n] > wl[end1] and wl[n] < wl[end2] and spec[n] > (np.mean(spec[range(end1 - 10, end1)]) + np.mean(spec[range(end2, end2 + 10)]))/2:
					spec[n] = (np.mean(spec[range(end1 - 10, end1)]) + np.mean(spec[range(end2, end2 + 10)]))/2
	#return the spectrum
	return spec

def make_reg(wl, flux, waverange):
	"""given some wavelength range as an array, output flux and wavelength vectors within that range.

	Args:
		wl (list): wavelength array
		flux (list): flux array
		waverange (list): wavelength range array

	Returns: 
		wavelength and flux vectors within the given range

	"""
	#find the smallest separation in the wavelength array

	#interpolate the input spectrum
	wl_interp = interp1d(wl, flux)
	#make a new wavelength array that's evenly spaced with the minimum spacing
	wlslice = np.arange(min(waverange), max(waverange), wl[1]-wl[0])
	#use the interpolation to get the evenly spaced flux
	fluxslice = wl_interp(wlslice)
	#return the new wavelength and flux
	return wlslice, fluxslice

def interp_2_spec(spec1, spec2, ep1, ep2, val):
	"""Args: 
		spec1 (list): first spectrum array (fluxes only)
		spec2 (list): second spectrum array (fluxes only)
		ep1 (float): First gridpoint of the value we want to interpolate to.
		ep2 (float): Second gridpoint of the value we want to interpolate to.
		val (float): a value between ep1 and ep2 that we wish to interpolate to.

	Returns: 
		a spectrum without a wavelength parameter

	"""	
	ret_arr = []
	#make sure the two spectra are the same length
	if len(spec1) == len(spec2):
		#go through the spectra
		for n in range(len(spec1)):
			#the new value is the first gridpoint plus the difference between them weighted by the spacing between the two gridpoints and the desired value.
			#this is a simple linear interpolation at each wavelength point
			v = ((spec2[n] - spec1[n])/(ep2 - ep1)) * (val - ep1) + spec1[n]
			ret_arr.append(v)
		#return the new interpolated flux array
		return ret_arr

	#otherwise yell at me because i'm trying to interpolate things that don't have the same length
	else:
		return('the spectra must have the same length')

def make_varied_param(init, sig):
	"""randomly varies a parameter within a gaussian range based on given std deviation

	Args:
		init (float): initial value
		sig (float): std deviation of gaussian to draw from

	Returns: 
		the varied parameter.

	"""
	var = np.random.normal(init, sig)
	
	return var

def find_model(temp, logg, metal):
	"""Finds a filename for a phoenix model with values that fall on a grid point.
	Assumes that model files are in a subdirectory of the working directory, with that subdirectory called "SPECTRA"
	and that the file names take the form "lte{temp}-{log g}-{metallicity}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
	The file should contain a flux column, where the flux is in units of log(erg/s/cm^2/cm/surface area). There should also be a
	wavelength file in the spectra directory called WAVE_PHOENIX-ACES-AGSS-COND-2011.fits, with wavelength in Angstroms.
	THE NEW SPECTRA are from Husser et al 2013.

	Args: 
		temperature (float): temperature value
		log(g) (float): log(g) value
		metallicity (float): Metallicity value

	Note:
		Values must fall on the grid points of the model grid. Only supports log(g) = 4 with current spectra directory.

	Returns: 
		file name of the phoenix model with the specified parameters.

	"""
	temp = str(int(temp)).zfill(5)
	metal = str(float(metal)).zfill(3)
	logg = str(float(logg)).zfill(3)
	file = glob('SPECTRA/lte{}-{}0-{}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits.txt'.format(temp, logg, metal))[0]
	return file

def spec_interpolator(trange, lgrange, specrange, npix = 3, resolution = 10000, metal = 0, write_file = True):
	'''Runs before emcee, to read in files to memory

	Args:
		trange (array): minimum and maximum temperature limits to be read in
		lgrange (array): minimum and maximum log(g) limits to be read in
		specrange (array): minimum and maximum wavelengths to use (in Angstroms)
		npix (int): factor by which to reduce the resolution. Default is 3
		resolution (int): resolution at which to store the spectra - should be larger than the final desired resolution
		metal (float): metallicity to use. Defaults to 0, which is also the only currently supported value.

	'''
	#first, read in the wavelength vector
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
			file = find_model(t[n], l[k], metal)
			#read it in
			with open(file, 'r') as f1:
				spec1 = []
				for line in f1:
					spec1.append(float(line))
				f1.close()
			#select the right wavelength region
			spec1 = np.array(spec1)[idx]

			#downsample - default is 3 pixels per resolution element
			res_element = np.mean(spwave)/resolution
			spec_spacing = spwave[1] - spwave[0]
			if npix * spec_spacing < res_element:
				factor = (res_element/spec_spacing)/npix
				wl, spec1 = redres(spwave, spec1, factor)

			#next, we just add it to the dictionary with the correct (temp, log(g)) tuple identifying it

			specs['{}, {}'.format(t[n], l[k])] = spec1

	specs['wl'] = wl

	return specs

def get_spec(temp, log_g, reg, specdict, metallicity = 0, normalize = False, wlunit = 'aa', pys = False, plot = False, model_dir = 'phoenix', resolution = 3000, reduce_res = True, npix = 3):
	"""Creates a spectrum from given parameters, either using the pysynphot utility from STScI or using a homemade interpolation scheme.
	Pysynphot may be slightly more reliable, but the homemade interpolation is more efficient (by a factor of ~2).
	
	TO DO: add a path variable so that this is more flexible, add contingency in the homemade interpolation for if metallicity is not zero

	Args: 
		temp (float): temperature value
		log_g (float): log(g) value
		reg (list): region array ([start, end])
		metallicity (float): Optional, defaults to 0
		normalize (boolean): Optional, defaults to True
		wlunit: Optional, wavelength unit. Defaults to angstroms ('aa'), also supports microns ('um').
		pys (boolean): Optional, set to True use pysynphot. Defaults to False.
		plot (boolean): Produces a plot of the output spectrum when it is a value in between the grid points and pys = False (defaults to False).
		resolution (int): Spectral resolution to broaden the spectrum to. Default is 3000.
		reduce_res (boolean): Whether to impose "pixellation" onto the spectrum using a designated number of pixels per resolution element. Default is True.
		npix (int): Number of pixels per resolution element if pixellating the spectrum.

	Returns: 
		a wavelength array and a flux array, in the specified units, as a tuple. Flux is in units of F_lambda (I think)

	Note:
		Uses the Phoenix models as the base for calculations. 

	"""
	if pys == True:
	#grabs a phoenix spectrum using Icat calls via pysynphot (from STScI) defaults to microns
	#get the spectrum
		sp = ps.Icat('phoenix', temp, metallicity, log_g)
		#put it in flambda units
		sp.convert('flam')
		#make arrays to eventually return so we don't have to deal with subroutines or other types of arrays
		spflux = np.array(sp.flux, dtype='float')
		spwave = np.array(sp.wave, dtype='float')

	if pys == False:
		#we have to:
		#read in the synthetic spectra
		#pick our temperature and log g values (assume metallicity is constant for now)
		#pull a spectrum 

		#initialize a time variable if i want to check how long this takes to run
		time1 = time.time()
		#list all the spectrum files
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

		#the spectra are in units of erg/s/cm^2/cm, so divide by 1e8 to get erg/s/cm^2/A, then multiply by stellar area to get a physical flux
		if lg1 == lg2 and temp1 == temp2:
			spflux = specdict['{}, {}'.format(temp1, lg1)]

		#If the points don't all fall on the grid points, we need to get the second spectrum at point t2 lg2, as well as the cross products
		#(t1 lg2, t2 lg1)
		else:
			#find the second file as well (i already found t1 lg1 before this if/else loop)
			spec1 = specdict['{}, {}'.format(temp1, lg1)]
			spec2 = specdict['{}, {}'.format(temp2, lg2)]
			t1_inter = specdict['{}, {}'.format(temp1, lg2)]
			t2_inter = specdict['{}, {}'.format(temp2, lg1)]


			# #and again for t2 lg1
			# f = find_model(temp2, lg1, 0)
			# with open(f, 'r') as t2:
			# 	t2_inter = []
			# 	for line in t2:
			# 		t2_inter.append(float(line))
			# 	t2.close()

			# for line in f:
			# 	l = line.strip().split(' ')
			# 	t2wave.append(l[0].strip())
			# 	if l[1] != '':
			# 		t2_inter.append(l[1].strip())
			# 	else:
			# 		t2_inter.append(l[2].strip())
			
			# t2wave = [float(w) for w in t2wave]
			# try:
			# 	t2_inter = [float(t2_inter[n]) for n in range(len(t2_inter))]
			# except:
			# 	print(find_model(temp2, lg1, 0))

			#so now I have four spectra, and I need to interpolate them correctly to get to some point between the grid points in both log(g) and teff space

			#make a new wl vector using the requested spectral region (which is given in microns, but we're working in angstroms) and that smallest wavelength step
			# wls = np.arange(min(reg)*1e4, max(reg)*1e4, wl1[1] - wl1[0])

			#Convert all the spectrafrom the weird PHOENIX units of log(s) + 8 to just s, where s is in units of erg/s/cm^2/A/surface area 
			spec1, spec2, t1_inter, t2_inter = [s/1e8 for s in spec1], [s/1e8 for s in spec2], \
				[s/1e8 for s in t1_inter], [s/1e8 for s in t2_inter]
			#interpolate everything onto the same grid using the newly defined wavelength array
			# iw1 = interp1d(wl1, spec1)
			# spec1 = iw1(wls)
			# iw2 = interp1d(wl2, spec2)
			# spec2 = iw2(wls)

			# it1 = interp1d(t1wave, t1_inter)
			# t1_inter = it1(wls)
			# it2 = interp1d(t2wave, t2_inter)
			# t2_inter = it2(wls)

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
	reg = [reg[n] * 1e4 for n in range(len(reg))]

	# #make sure the flux array is a float not a string
	spflux = [float(s) for s in spflux]
	#and truncate the wavelength and flux vectors to contain only the requested region
	spwave, spflux = make_reg(spwave, spflux, reg)
	#you can choose to normalize
	if normalize == True:
		spflux /= max(spflux)

	#this is the second time object in case you want to check runtime
	# print('runtime for spectral retrieval (s): ', time.time() - time1)
	#broaden the spectrum to mimic the dispersion of a spectrograph using the input resolution
	spwave, spflux = broaden(spwave, spflux, resolution)

	#and reduce the resolution again to mimic pixellation from a CCD
	#i should make this nicer - it currently assumes you want three resolution elements per angstrom, which isn't necessarily true
	if reduce_res == True:
		res_element = np.mean(spwave)/resolution
		spec_spacing = spwave[1] - spwave[0]
		if npix * spec_spacing < res_element:
			factor = (res_element/spec_spacing)/npix
			spwave, spflux = redres(spwave, spflux, factor)

	#depending on the requested return wavelength unit, do that, then return wavelength and flux as a tuple
	if wlunit == 'aa': #return in angstroms
		return spwave, spflux
	elif wlunit == 'um':
		spwave = spwave * 1e-4
		return spwave, spflux
	else:
		factor = float(input('That unit is not recognized for the return unit. \
			Please enter a multiplicative conversion factor to angstroms from your unit. For example, to convert to microns you would enter 1e-4.'))
		spwave = [s * factor for s in spwave]

		return spwave, spflux

def add_spec(teff, logg, frs, filts, r, specs, normalize = True, mode = 'spec'):#, waverange):
	"""add spectra together given an array of spectra and flux ratios

	Args: 
		teff (array): array of temperature values (floats)
		logg (array): array of log(g) values (floats) 
		frs (array): Array of flux ratios. if there are more than two stars in a system, enter each filter's flux ratio as a space-separated string, 
		where the first entry is the primary:secondary, second string is the primary:tertiary, etc. E.g., ['0.5 0.3', '0.3 0.1']
		filts (array): Array of strings specifying which filters to use to normalize with. \
		Supported systems are 2MASS, Bessell, Cousins, Johnson, Landolt, SDSS, and Stromgren.\
		For standard UBVRIJHK defaults to Johnson (UBVRI) and Bessell (JHK). Should be entered as, e.g., 'Johnson, U'. Not case sensitive.
		range (array): minimum and maximum limits for the spectrum, in units of microns
		normalize (boolean): Normalize the spectra before adding them (default is True)
		mode (string): use either photometry ('SED') or spectra + flux ratio(s) ('spec') to return.

	Returns: 
		wl, spec (tuple): wavelength and spectrum (or central wavelength and synthetic photometry) for a spectrum 

	"""
	if not len(frs) == len(filts):
		return "Number of flux ratios must be equal to the number of filters!"

	#initialize variables for keeping track of the total minimum and maximum wavelengths requested for all filters
	tmi, tma = np.inf, 0

	#and initialize lists for keeping track of all the transmission curves and wavelength ranges for all filters
	wls, tras = [], []
	
	#go through the filters and get all the information from the input string
	for f in filts:
		#first, we have to figure out what filter this is
		#make sure it's lowercase
		f = f.lower()
		#get the system and filter from the input string
		try:
			if ',' in f:
				sys, fil = f.split(','); sys = sys.strip(); fil = fil.strip()
			else:
				fil = f.strip()
				if f in 'ubvri':
					sys = 'johnson'
				elif f in 'jhk':
					sys = 'bessell'

		except:
			print('Please format your filter as, e.g., "Johnson, V". The input is case insensitive.')
			break

		#now get the fits file version of the transmission curve from the "bps" directory
		#which should be in the same directory as the code
		filtfile = fits.open('bps/{}_{}_002.fits'.format(sys, fil))[1].data

		t_wl, t_cv = filtfile['WAVELENGTH'], filtfile['THROUGHPUT']
		#record the wavelength and transmission curve for each filter outside of the loop
		wls.append(t_wl); tras.append(t_cv)

		#get the wavelength range
		t_min, t_max = min(t_wl), max(t_wl)

		#and check if it's either the min or max requested - this will define the wavelength for spectral retrieval
		if t_min < tmi:
			tmi = t_min 
		if t_max > tma:
			tma = t_max

	#get the primary star wavelength array and spectrum 
	#get an R ~ 10,000 spectrum so that the error is small
	#the min and max wavelength points will be in Angstroms so we need to make them microns for the function call
	pri_wl, pri_spec = get_spec(teff[0], logg[0], [min(min(r), tmi/1e4) - 5e-4, max(max(r), tma/1e4) + 5e-4], specs, normalize = True, resolution = 10000)

	#initialize a fancy plot 
	#the first row will be the "observed" spectrum for each filter, while the second row will be the original primary star spectrum
	#with the transmission curves overlaid as color
	# fig, [ax1, ax2] = plt.subplots(nrows = 2, sharex = True)

	# for n in range(len(filts)):
	# 	ax2.axvspan(min(wls[n]), max(wls[n]), label = filts[n], alpha = 0.5)
	# ax2.plot(pri_wl, pri_spec)
	# ax2.legend()

	#now we need to get the secondary (and possibly higher-order multiple) spectra
	#given the way the spectral retrieval code works, as long as the wavelength range is the same the spectra will be on the same grid
	#so I'm just going to stack them with the primary wavelength and spectrum - going to save the wavelength just in case I need to troubleshoot later
	for n in range(1, len(teff)):
		sec_wl, sec_spec = get_spec(teff[n], logg[n], [min(min(r), tmi/1e4) - 5e-4, max(max(r), tma/1e4) + 5e-4], specs, normalize = True, resolution = 10000)

		pri_wl = np.row_stack((pri_wl, sec_wl)); pri_spec = np.row_stack((pri_spec, sec_spec))

	#define an array to hold all my "instrumental" fluxes
	mags = np.zeros((len(filts), len(teff)))

	#loop through each filter 
	for n in range(len(filts)):
		#and get the wavelength range and transmission curve
		ran, tm = wls[n], tras[n]
		#pick out the region of stellar spectrum that matches the curve
		w = pri_wl[0][np.where((pri_wl[0] < max(ran)) & (pri_wl[0] >= min(ran)))]
		#and interpolate the transmission curve so that it matches the stellar spectral resolution
		intep = interp1d(ran, tm)
		tran = intep(w)
		#now for each star
		for k in range(len(teff)):
			#pick out the spectrum over the appropriate region
			s = pri_spec[k][np.where((pri_wl[0] < max(ran)) & (pri_wl[0] >= min(ran)))]
			t_spec = [s[p] * tran[p] for p in range(len(s))]
			# ax1.plot(w, t_spec, label = filts[n])
			#put it through the filter by multiplying by the transmission, then integrate to finally get the instrumental flux
			m = np.trapz(t_spec, w)
			#and add it to the array in the appropriate place for this star and filter
			mags[n][k] = m
	# ax1.legend()
	# plt.tight_layout()
	# plt.savefig('spectrum_check.png')

	if mode == 'sed' or mode == 'SED':
		#Need to multiply each instrumental flux by the correct flux ratio, then add them to get an unresolved magnitude
		#develop this fully later
		pass
	elif mode == 'spec':
		#don't allow use with more than one filter
		if len(filts) > 1:
			print('Can\'t use spectral mode with more than 1 flux ratio')
			return

		# fig, ax = plt.subplots()
		# ax.plot(pri_wl[0], pri_spec[0], label = 'primary')
		#first, measure the "intrinsic" flux ratio
		for n in range(1, len(teff)):
			init_fr = mags[0][n]/mags[0][0]
			#and read in the flux ratio from the given array
			#if there are more than two stars, do some string comprehension to find the right number
			try:
				new_fr = float(frs[0].split(' ')[n - 1])
			#if that doesn't work (because there's no space in the string), just read in the single value - this is what will happen for a binary
			except:
				new_fr = float(frs[0])

			#now, take the ratio of the two: We want to divide the secondary spectrum by the intrinsic flux ratio (moving it to a value of 1), then divide by the 
			#assigned flux ratio
			#this gives it the correct normalization using the measured flux ratio
			pri_spec[n] = [s * new_fr/init_fr for s in pri_spec[n]]

		spec1 = pri_spec[0]
		for n in range(1, len(teff)):
			spec1 += pri_spec[n]

		# #I also want to plot the two newly normalized spectra, the two flux ratios, and the non-normalized final composite spectrum 

		# ax.plot(pri_wl[1], pri_spec[1], label = 'secondary')
		# ax.plot(pri_wl[0], spec1, alpha = 0.5, label = 'composite')
		# ax.plot(pri_wl[0], spec1/max(spec1), label = 'normalized composite')
		# ax.scatter([np.mean(wls), np.mean(wls), np.mean(wls)], [mags[0][0]/mags[0][0], mags[0][1]/mags[0][0], np.sum(mags[0])/mags[0][0]], label = 'synth. phot.')
		# ax.set_xlabel('wavelength (A)')
		# ax.set_ylabel('relative flux')
		# ax.legend(loc = 'best')
		# plt.tight_layout()
		# plt.savefig('synth_phot_check.png')

	if normalize == True:
		spec1/= max(spec1)

	return pri_wl[0], spec1

def make_bb_continuum(wl, spec, dust_arr, wl_unit = 'um'):
	"""Adds a dust continuum to an input spectrum.

	Args:
		wl (list): wavelength array
		spec (list): spectrum array
		dust_arr (list): an array of dust temperatures
		wl_unit (string): wavelength unit - supports 'aa' or 'um'. Default is 'um'.

	Returns:
		a spectrum array with dust continuum values added to the flux.

	"""
	h = 6.6261e-34 #J * s
	c = 2.998e8 #m/s
	kb = 1.3806e-23 # J/K

	if wl_unit == 'um':
		wl = [wl[n] * 1e-6 for n in range(len(wl))] #convert to meters
	if wl_unit == 'aa':
		wl = [wl[n] * 1e-10 for n in range(len(wl))]

	if type(dust_arr) == float or type(dust_arr) == int:
		pl = [(2 * h * c**2) /((wl[n]**5) * (np.exp((h*c)/(wl[n] * kb * dust_arr)) - 1)) for n in range(len(wl))]

	if type(dust_arr) == np.isarray():
		for temp in dust_arr:
			pl = [(2 * h * c**2) /((wl[n]**5) * (np.exp((h*c)/(wl[n] * kb * temp)) - 1)) for n in range(len(wl))]

			spec = [spec[n] + pl[n] for n in range(len(pl))]
	return spec

def fit_prior(temps, lgs, extinct, t_prior, lg_prior, ext_prior):
	if extinct < 0 or extinct > 5 or any(t > 5000 for t in temps) or any(t < 2300 for t in temps) or any(l < 3 for l in lgs) or any(l > 5 for l in lgs)\
	or temps[0] < temps[1]:
		return -np.inf

	else:
		return 0

def fit_spec(n_walkers, wl, flux, reg, t_guess, lg_guess, extinct_guess, fr_guess, metal_guess = 0, dust_guess = 0, wu='aa',\
 burn = 100, cs = 10, steps = 200, dust = False, pysyn = False, conv = True):
	"""Does an MCMC to fit a combined model spectrum to an observed single spectrum.
	guess_init and sig_init should be dictionaries of component names and values for the input guess and the 
	prior standard deviation, respectively. 
	Assumes they have the same metallicity.
	The code will expect an dictionary with values for temperature ('t'), log g ('lg'), and dust ('dust') right now.
	TO DO: add line broadening, disk/dust to possible fit params.

	Args:
		n_walkers (int): number of walkers
		wl (list): wavelength array
		flux (list): spectrum array
		reg (list): Two value array with start and end points for fitting.
		fr (list): flux ratio array. Value1 is flux ratio, value2 is location in the spectrum of value1, etc.
		guess_init (dictionary): dictionary of component names and values for the input guess. \
			The code will expect an dictionary with values for temperature ('t'), log g ('lg'), extinction ('extinct'), and dust ('dust').
		sig_init (dictionary): A dictionary with corresponding standard deviations for each input guess. Default is 200 for temperature, 0.2 for log(g)
		wu (string): wavelength unit. currently supports 'aa' or 'um'. Default: "um".
		burn (int): how many initial steps to discard to make sure walkers are spread out. Default: 100.
		cs (int): cutoff chi square to decide convergence. Default: 10.
		steps (int): maximum steps to take after the burn-in steps. Default: 200.
		pysyn (Bool): Boolean command of whether or not to use pysynphot for spectral synthesis. Default: False
		conv (Bool): Use chi-square for convergence (True) or the number of steps (False). Default: True.
		dust (Bool): Add a dust spectrum. Default: False.

	"""
	wl *= 1e4
	if metal_guess > 0:
		metal = metal_guess
	else:
		metal = 0

	# #make some initial guess' primary and secondary spectra, then add them
	# wave1, spec1 = get_spec(t_guess[0], lg_guess[0], [reg[0] - 0.005, reg[1] + 0.005], metallicity = metal, wlunit = wu, normalize = True, resolution = 10000)
	# wave2, spec2 = get_spec(t_guess[1], lg_guess[1], [reg[0] - 0.005, reg[1] + 0.005], metallicity = metal, wlunit = wu, normalize = True, resolution = 10000)

	# intep = interp1d(wave2, spec2)
	# spec2 = intep(wave1)

	wave1, init_cspec = add_spec(t_guess, lg_guess, fr_guess[0], fr_guess[1], reg)

	init_cspec = extinct(wave1, init_cspec, extinct_guess)

	if dust_guess > 0:
		init_cspec = add_dust(init_cspec, dust_guess)

	intep = interp1d(wave1, init_cspec)
	init_cspec = intep(wl)

	#calculate the chi square value of that fit
	ic = chisq(flux, init_cspec, np.std(flux))
	init_cs = np.sum(ic)/len(ic)
	#that becomes your comparison chi square
	chi = init_cs
	#make a random seed based on your number of walkers
	np.random.seed(n_walkers + np.random.randint(2000))

	#savechi will hang on to the chi square value of each fit
	savechi = []

	#sp will hang on to the tested set of parameters at the end of each iteration
	sp = [t_guess, lg_guess, [extinct_guess], [dust_guess]]

	si = [[200, 200], [0.1, 0.1], [0.05], [0]]
	var_par = sp
	gi = sp

	n = 0
	#print('Starting MCMC walker {}....(this might take a while)'.format(n_walkers + 1))
	while n < steps + burn:
		vp = np.random.randint(0, len(var_par))
		var_par[vp] = make_varied_param(var_par[vp], si[vp])

		if var_par[2][0] > 0 and var_par[0][0] > 2000 and var_par[0][0] < 6000 and var_par[0][1] > 2000 and var_par[0][1] < 6000\
		and var_par[1][0] > 3 and var_par[1][0] < 5 and var_par[1][1] > 3 and var_par[1][1] < 5 and var_par[0][1] < var_par[0][0]:

			#make spectrum from varied parameters
			# test_wave1, test_spec1 = get_spec(var_par[0][0], var_par[1][0], [reg[0] - 0.005, reg[1] + 0.005], wlunit = wu, resolution = 10000)
			# test_wave2, test_spec2 = get_spec(var_par[0][1], var_par[1][1], [reg[0] - 0.005, reg[1] + 0.005], wlunit = wu, resolution = 10000)

			# intep = interp1d(test_wave2, test_spec2)
			# test_spec2 = intep(test_wave1)

			# test_cspec = add_spec([test_wave1, test_wave1], [test_spec1, test_spec2], var_par[2])

			test_wave1, test_cspec = add_spec(var_par[0], var_par[1], fr_guess[0], fr_guess[1], reg)


			if dust == True:
				test_cspec = add_dust(test_cspec, var_par[3])

			test_cspec = extinct(test_wave1, test_cspec, var_par[2][0])

			intep = interp1d(test_wave1, test_cspec)
			test_cspec = intep(wl)

			#calc chi square between data and proposed change
			tc = chisq(test_cspec, flux, np.std(flux))
			test_cs = np.sum(tc)/len(tc)

		else:
			vp = np.random.randint(0, len(var_par))
			var_par[vp] = make_varied_param(var_par[vp], si[vp])

		lh = test_cs

		u = np.random.uniform(0, 1)

		if test_cs/chi > u:
			gi[vp] = var_par[vp]
			chi = test_cs 

		f = open('results/params{}.txt'.format(n_walkers), 'a')
		f.write('{} {} {} {} {} {}\n'.format(list(gi[0])[0], list(gi[0])[1], list(gi[1])[0], list(gi[1])[0], float(gi[2][0]), float(gi[3][0])))
		f.close()
		f = open('results/chisq{}.txt'.format(n_walkers), 'a')
		f.write('{} {}\n'.format(chi, test_cs))
		f.close()

		# sp = np.vstack((sp, gi))
		savechi.append(chi)
		if conv == True and n > burn:
			if savechi[-1] <= cs:
				n = steps + burn
				print("Walker {} is done.".format(n_walkers + 1))
			elif savechi[-1] > cs:
				n = burn + 5
			else:
				print('something\'s happening')
		else:
			n += 1

	return savechi[-1]

def run_mcmc(walk, w, flux, regg, values, fr, steps = 200, burn = 100, chi = 10, conv = True):
	#use multiple walkers and parallel processing:

	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()

	wl = np.arange(0, walk)

	walker_num = comm.scatter(wl, root = 0)

	t1, t2, lg1, lg2, extinct = values[walker_num]

	results = fit_spec(walker_num, w, flux, regg, [t1, t2], [lg1, lg2], extinct, fr, burn = burn, steps = steps, cs = chi, conv = conv)
	out = comm.gather(results, root = 0)

	#print('Writing file')
	# np.savetxt('results/multi_walkers.txt', out, fmt = '%.8f')

	return

def loglikelihood(p0, fr, nspec, ndust, data, broadening, r, specs, w = 'aa', pysyn = False, dust = False, norm = True, mode = 'spec'):
	"""The natural logarithm of the joint likelihood. 
	Set to the chisquare value. (we want uniform acceptance weighted by the significance)
	
	Possible kwargs are reg (region), wlunit ('um' or 'aa'), dust (defaults to False), \
		normalize (defaults to True), pysyn (defaults to False), 

	Args:
		p0 (list): a sample containing individual parameter values. Then p0[0: n] = temp, p0[n : 2n] = lg, p0[2n : -1] = dust temps
		nspec (int): number of spectra/stars
		ndust (int): number of dust continuum components
		data (list): the set of data/observations
		fr (array): must contain two arrays: first, a list of flux ratios, \
			second, a list of filter names, with system and wavelength separated by a comma (e.g., 'cousins, r').
		broadening (int): The instrumental resolution of the spectra
		r (list): region to use when calculating liklihood
		w (string): Wavelength unit, options are "aa" and "um". Default is "aa".
		pysyn (bool): Use pysynphot to calculate spectra. Default is False.
		dust (bool): Add dust continuum? Default is False.
		norm (bool): Normalize spectra when fitting. Default is True.
		mode (string): do calculation using only synthetic photometry ('SED') or a spectrum with flux ratio(s) ('spec')

	Returns: 
		cs (float): a reduced chi square value corresponding to the quality of the fit.

	Note:
		current options: arbitrary stars, dust (multi-valued). 
		To do: fit for broadening or vsini.

	"""
	# le = len(data)

	# wl = np.zeros(le)
	# spec = np.zeros(le)

	#print('getting spectra')
	# for n in range(nspec):
	# 	if len(p0) == nspec:
	# 		lg = 4
	# 	else:
	# 		lg = p0[nspec + n]

	# 	ww, spex = get_spec(p0[n], lg, normalize = norm, reg = r, wlunit = w, pys = pysyn)

	# 	wl1 = np.linspace(min(ww), max(ww), le)

	# 	if len(spex) == 0:
	# 		spex = np.ones(len(ww))

	# 	#print(le, np.shape(ww), np.shape(spex))
	# 	intep = scipy.interpolate.interp1d(ww, spex)
	# 	spec1 = intep(wl1)

	# 	wl = np.vstack((wl, wl1))
	# 	spec = np.vstack((spec, spec1))

	# if len(fr) == 1:
	# 	test_spec = add_spec(wl, spec, [fr, np.mean(wl)])
	# else:
	# 	test_spec = add_spec(wl, spec, fr)

	#print('creating composite spectrum')
	wl, spec = add_spec(p0[0:nspec], p0[nspec:2*nspec], fr[0], fr[1], r, specs, mode = mode)

	# if mode == 'spec':
	# 	wl, spec = wl[np.where((wl > min(r)*1e4) & (wl < max(r)*1e4))], spec[np.where((wl > min(r)*1e4) & (wl < max(r)*1e4))]

	test_spec = extinct(wl, spec, p0[nspec * 2])

	if dust == True:
		test_spec = make_bb_continuum([wl, test_spec], p0[2 * nspec + 1: -1], wl_unit = w)

	test_wl, test_spec = broaden(wl, test_spec, broadening)

	test_wl /= 1e4
	intep = interp1d(test_wl, test_spec)

	w, s = data
	it = intep(w)

	ic = chisq(it, s, np.std(s))

	init_cs = np.sum(ic)/len(ic)

	if np.isnan(init_cs):
		return -np.inf
	else:
		return -0.5 * init_cs

def logprior(p0, nspec, ndust, prior = 0):
	temps = p0[0:nspec]
	lgs = p0[nspec:2 * nspec]
	extinct = p0[2*nspec]

	if ndust > 0:
		dust = p0[2 * nspec + 1:]
	
	if extinct < 0 or extinct > 5 or any(t > 5000 for t in temps) or any(t < 3300 for t in temps) or any(l < 3.5 for l in lgs) or any(l > 4.5 for l in lgs)\
	or temps[0] < temps[1]:
		return -np.inf
	elif prior != 0:
		tprior = prior[:nspec]
		tpsig = prior[nspec:2*nspec]
		lprior = prior[2*nspec:3*nspec]
		lpsig = prior[3*nspec:4*nspec]
		eprior = prior[4*nspec]
		epsig = prior[4*nspec+1]

		pp = []
		ps = tprior + lprior + [eprior]
		ss = tpsig + lpsig + [epsig]

		for k, p in enumerate(ps):
			if p != 0:
				like = -0.5 * ((p0[k] - p)/ss[k])**2
				pp.append(like)

		return np.sum(pp)

	else:
		print('either set your prior to zero or enter an array with [tprior1, tprior2, tsig1, tsig2...] for temp, log(g), and extinction')

def logposterior(p0, fr, nspec, ndust, data, broadening, r, specs, wu = 'aa', pysyn = False, dust = False, norm = True, prior = 0):
	"""The natural logarithm of the joint posterior.

	Args:
		p0 (list): a sample containing individual parameter values. Then p0[0: n] = temp, p0[n : 2n] = lg, p0[2n+1] = flux ratio,\
			 p0[2n + 2] = extinction, p0[2n+3:-1] = dust temps
		nspec (int): number of spectra/stars
		ndust (int): number of dust continuum components
		data (list): the set of data/observations
		flux_ratio (array): set of flux ratios with corresponding wavelength value for location of ratio
		broadening (int): The instrumental resolution of the spectra
		r (list): region to use when calculating liklihood
		w (string): Wavelength unit, options are "aa" and "um". Default is "aa".
		pysyn (bool): Use pysynphot to calculate spectra. Default is False.
		dust (bool): Add dust continuum? Default is False.
		norm (bool): Normalize spectra when fitting. Default is True.

	Returns: 
		lh (float): The log of the liklihood of the fit being pulled from the model distribution.

	Note:
		Assuming a uniform prior for now

	"""
	t1 = time.time()
	lp = logprior(p0, nspec, ndust, prior = prior)
	# if the prior is not finite return a probability of zero (log probability of -inf)
	if not np.isfinite(lp):
		return -np.inf

	else:
		lh = loglikelihood(p0, fr, nspec, ndust, data, broadening, r, specs, w = wu, pysyn = False, dust = False, norm = True)
		# return the likeihood times the prior (log likelihood plus the log prior)
		print('time for likelihood call:', time.time() - t1)
		return lp + lh

def run_emcee(fname, nwalkers, nsteps, ndim, nburn, pos, fr, nspec, ndust, data, broadening, r, specs, nthin=10, w = 'aa', pys = False, du = False, no = True, prior = 0):
	"""Run the emcee code to fit a spectrum 

	Args:
		fname (string): input file name to use
		nwalkers (int): number of walkers to use
		nsteps (int): number of steps for each walker to take
		ndim (int): number of dimensions to fit to. For a single spectrum to fit temperature and log(g) for, ndim would be 2, for example. 
		nburn (int): number of steps to discard before starting the sampling. Should be large enough that the walkers are well distributed before sampling starts.
		pos (list): array containing the initial guesses for temperature, log g, flux ratio, and extinction
		nspec (int): number of spectra to fit to. For a single spectrum fit this would be 1, for a two component fit this should be 2.
		ndust (int): number of dust continuum components to fit to. (untested)
		data (list): the spectrum to fit to
		flux_ratio (list): an array with a subarray of flux ratios, followed by a subarray with the strings of the filter in which they were measured.
		broadening (float): the instrumental resolution of the input data, or the desired resolution to use to fit.
		r (list): a two valued array containing the region to fit within, in microns.
		nthin (int): the sampling rate of walker steps to save. Default is 10.
		w (string): the wavelength unit to use. Accepts 'um' and 'aa'. Default is 'aa'.
		pys (boolean): Whether to use pysynphot for spectral synthesis (if true). Default is False.
		du (boolean): Whether to fit to dust components. Default is False.
		no (boolean): Whether to normalize the spectra while fitting. Default is True.
	
	Note:
		This is still in active development and doesn't always work.

	"""

	# if which == 'pt':
	# 	ntemps = int(input('How many temperatures would you like to try? '))
	# 	sampler = emcee.PTSampler(ntemps, nwalkers, ndim, loglikelihood, logprior, threads=nwalkers, loglargs=[\
	# 	nspec, ndust, data, flux_ratio, broadening, r], logpargs=[nspec, ndust], loglkwargs={'w':w, 'pysyn': pys, 'dust': du, 'norm':no})

	# 	for p, lnprob, lnlike in sampler.sample(pos, iterations=nburn):
	# 		pass
	# 	sampler.reset()

	# 	#for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,lnlike0=lnlike, iterations=nsteps, thin=nthin):
	# 	#	pass

	# 	assert sampler.chain.shape == (ntemps, nwalkers, nsteps/nthin, ndim)

	# 	# Chain has shape (ntemps, nwalkers, nsteps, ndim)
	# 	# Zero temperature mean:
	# 	mu0 = np.mean(np.mean(sampler.chain[0,...], axis=0), axis=0)

	# 	try:
	# 		# Longest autocorrelation length (over any temperature)
	# 		max_acl = np.max(sampler.acor)
	# 		print('max acl: ', max_acl)
	# 		np.savetxt('results/acor.txt', sampler.acor)
	# 	except:
	# 		pass

	# if which == 'em':

	with MPIPool() as pool:
		if not pool.is_master():
			pool.wait()
			sys.exit(0)
		sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, threads=nwalkers, args=[fr, nspec, ndust, data, broadening, r, specs], \
		kwargs={'pysyn': pys, 'dust': du, 'norm':no, 'prior':prior}, pool = pool)
		
		for n, s in enumerate(sampler.sample(pos, iterations = nburn)):
			with open('results/{}_{}_burnin.txt'.format(fname, n), 'ab') as f:
				f.write(b"\n")
				np.savetxt(f, s.coords)
				f.close() 
			#f = open('results/{}_burnin.txt'.format(fname), "a")
			#f.write(s.coords)
			#f.close()
		state = sampler.get_last_sample()
		sampler.reset()
		old_acl = np.inf
		for n, s in enumerate(sampler.sample(state, iterations = nsteps)):
			with open('results/{}_{}_results.txt'.format(fname, n), 'ab') as f:
				f.write(b'\n')
				np.savetxt(f, s.coords)
				f.close()
			acl = sampler.get_autocorr_time(tol = 0)

			macl = np.mean(acl)
			
			with open('results/{}_autocorr.txt'.format(fname), 'a') as f:
				f.write(str(macl) + '\n')
				f.close()
			
			if not np.isnan(macl):
				converged = np.all(acl * 100 < sampler.iteration)
				converged &= np.all(np.abs(old_acl - acl) / acl < 0.01)
				if converged:
					break
			old_acl = acl
			#f = open('results/{}_results.txt'.format(fname), 'a')
			#f.write(s.coords)
			#f.close()
		#np.savetxt('results/{}_results.txt'.format(fname), sampler.flatchain)

	print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

	for i in range(ndim):
		plt.figure(i)
		plt.hist(sampler.flatchain[:,i], nsteps, histtype="step")
		plt.title("Dimension {0:d}".format(i))
		plt.savefig(os.getcwd() + '/results/plots/{}_{}.pdf'.format(fname, i))
		plt.close()

		plt.figure(i)

		try:
			for n in range(nwalkers):
				plt.plot(np.arange(nsteps),sampler.chain[n, :, i])
			plt.savefig(os.getcwd() + '/results/plots/{}_chain_{}.pdf'.format(fname, i))
			plt.close()
		except:
			pass

	samples = sampler.chain[:, :, :].reshape((-1, ndim))
	fig = corner.corner(samples, labels = ['Temp1', 'Temp2', 'Log(g)1', 'Log(g)2', 'Av'], show_titles = True)
	fig.savefig(os.getcwd() + "/results/plots/{}_corner.pdf".format(fname))
	plt.close()
	return

def plot_emcee(dirname, runname, fr, specs):
	files = glob(dirname + '/' + runname + '*results.txt')
	numbers = [int(f.split('_')[2]) for f in files]
	num = max(numbers)

	results = np.genfromtxt(dirname + '/{}_{}_results.txt'.format(runname, num))

	figure = corner.corner(results[:, :].reshape((-1, 5)), labels = ['Temp1', 'Temp2', 'Log(g)1', 'Log(g)2', 'Av'], show_titles = True)
	figure.savefig('results/plots/triangle.pdf')

	tem1, tem2, log1, log2, ext = [], [], [], [], []

	for line in results:
			tem1.append(line[0]); tem2.append(line[1])
			log1.append(line[2]); log2.append(line[3])
			ext.append(line[-1])

	tt1, tt2 = np.median(tem1), np.median(tem2); tl1, tl2 = np.median(log1), np.median(log2)
	te = np.median(ext)

	ww, flux = np.genfromtxt('Data/synth_spec.txt', unpack = True)

	ww = ww[1:-1]
	flux = flux[1:-1]
	cen_wl = ww[int(len(ww)/2)]
	regg = [min(ww)/1e4, max(ww)/1e4]

	wl1, spec = add_spec([tt1, tt2], [tl1, tl2], fr[0], fr[1], regg, specs)

	spec = extinct(wl1, spec, te)

	fig, ax = plt.subplots()
	# ax.plot(wl1, spec1, label = 'Primary: Teff = {:.0f}'.format(tt1))
	# ax.plot(wl2, spec2 * fr[0], label = 'Secondary: Teff = {:.0f}'.format(tt2))
	ax.plot(wl1, spec, label = 'Composite spectrum: {:.0f} + {:.0f}; {:.1f} + {:.1f}; {:.2f}'.format(tt1, tt2, tl1, tl2, te))
	ax.plot(ww, flux, label = 'Synthetic spectrum: 4250 + 3825; 4.2 + 4.3; 0.35')
	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel('Wavelength (A)', fontsize = 13)
	ax.set_ylabel('Normalized flux', fontsize = 13)
	ax.legend(fontsize = 13, loc = 'best')
	plt.tight_layout()
	plt.savefig(dirname + '/plots/spec_plot.png')
	plt.close()

def main():
	data_wl, data_spec = np.genfromtxt('Data/synth_spec.txt', unpack = True)#np.genfromtxt('Data/Spectra_for_Kendall/fftau_wifes_spec.csv', delimiter=',', unpack = True)

	data_wl = data_wl[1:-1]/1e4
	data_spec = data_spec[1:-1]

	t1 = time.time()
	specs = spec_interpolator([3300, 5000], [3.5,4.5], [3000, 10000], resolution = 15000)
	print('time to read in specs:', time.time() - t1)
	# fig, ax = plt.subplots(2, sharex = True)
	# ax[0].plot(data_wl, data_spec, label = "old", color='g')

	newspec = rmlines(data_wl, data_spec)

	# ax[1].plot(data_wl, newspec, label="trimmed", color='k')
	# fig.legend()
	# plt.savefig('modified_spec.pdf')

	data = [data_wl, newspec]

	nspec, ndust = 2, 0

	nwalkers, nsteps, ndim, nburn, broadening = 12, 2000, 5, 1, 3000

	# a = mft.run_mcmc(nwalkers, data_wl, data_spec, [min(data_wl), max(data_wl)], [4500,3200], [4, 4])

	#t1, t2, log(g)1, log(g)2, extinction
	pos = [4100, 3700, 4, 4, 2]
	#
	p0 = emcee.utils.sample_ball(pos, [300, 300, 0.1, 0.1, 0.2], size=nwalkers)

	# run_mcmc(nwalkers, data_wl, newspec, [min(data_wl), max(data_wl)], p0, [[0.4], ['johnson, r']], steps = nsteps, burn = nburn, conv = True, chi = 1.5)
	a = run_emcee('run1_small', nwalkers, nsteps, ndim, nburn, p0, [[0.4], ['johnson, r']], nspec, ndust, data, broadening, [min(data_wl), max(data_wl)], specs,\
		nthin=2, w = 'aa', pys = False, du = False, prior = [4100, 3700, 300, 300, 4, 4, 0.2, 0.2, 2, 0.05])

	plot_emcee('results', 'run1_small', [[0.4], ['johnson, r']], specs)

if __name__ == "__main__":
	main()