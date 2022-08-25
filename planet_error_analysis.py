import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.units as u
from astropy.io import ascii
from glob import glob 
from sklearn.neighbors import KernelDensity
from scipy.optimize import curve_fit
from matplotlib import ticker

def max_greenhouse_limit(tstar): 
	ts = tstar - 5780 
	return 0.356 + (6.171e-5)*ts + (1.698e-9)*ts**2 - (3.198e-12)*ts**3 - (5.575e-16)* ts**4

def recent_venus(tstar):
	ts = tstar - 5780
	return 1.776 + (2.136e-4)*ts + (2.533e-8)*ts**2 - (1.332e-11)*ts**3 - (3.097e-15)* ts**4

def runaway_greenhouse(tstar):
	ts = tstar - 5780
	return 1.107 + (1.332e-4 )*ts + (1.58e-8)*ts**2 - (8.308e-12)*ts**3 - (1.931e-15)* ts**4

tstars = np.arange(2700, 7200, 50)

all_koi_table = Table(ascii.read('targets/targets_kep/all_kois.csv'))
tt = [t.split('i')[1] for t in glob('koi*')]
targets = []
for t in tt:
	if len(t) <= 4:
		targets.append(t)

targets.sort()

kois = np.array(all_koi_table['KOI'])

target_planets = []
for t in targets:
	koi_planets = kois[np.where([int(k) == int(t) for k in kois])]
	for k in koi_planets:
		target_planets.append(k)

target_table = all_koi_table[np.where([all_koi_table['KOI'] == t for t in target_planets])[1]]
target_table['Period error'] /= 365.25
target_table['Period (days)'] /= 365.25

columns = ['kic','pname', 'radius', 'radius_err', 'teq', 'period', 'period_err', 's', 's_err', 'rp/rstar', 'rp/rstar error']
planet_table = Table(target_table['KIC ID','KOI', 'Radius (R_Earth)','Radius error','Eq Temp (K)','Period (days)','Period error','Insolation (Earth flux)','Insolation error', 'Planet Radius/Stellar Radius', 'Planet Rad/Stellar Rad error'], names = columns, dtype = ['S','S', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f','f'])
planet_table['tstar_index'] = np.zeros(len(target_table))

kic_nonredundant = []
for k in list(planet_table['kic']):
	if k not in kic_nonredundant:
		kic_nonredundant.append(k)

star_params = Table(ascii.read('Rgap/exoarchive_kepler_stellar_params.tsv'))
star_target_params = star_params[np.where([star_params['kepid'] == int(k) for k in kic_nonredundant])[1]]

star_target_params['koi'] = targets

for n in range(len(star_target_params)):
	star_target_params['koi'][n] = star_target_params['koi'][n].split('.')[0]

furlan_prcf = Table(ascii.read('Rgap/furlan2017_prcf.tsv'))

prcfs = furlan_prcf[np.where([furlan_prcf['KOI'] == t for t in targets])[1]]

primary_prcfs = prcfs[np.where([prcfs['Orbit'] == 'primary  '])[1]]
secondary_prcfs = prcfs[np.where([prcfs['Orbit'] == 'companion'])[1]]

furlan_primary_prcf = []; furlan_secondary_prcf = []

for t in targets:
	if t in primary_prcfs['KOI']:
		furlan_primary_prcf.append(float(primary_prcfs['Avg'][np.where(primary_prcfs['KOI'] == t)]))
		furlan_secondary_prcf.append(float(secondary_prcfs['Avg'][np.where(secondary_prcfs['KOI'] == t)]))
	else:
		furlan_primary_prcf.append(np.nan)
		furlan_secondary_prcf.append(np.nan)

furlan_sep = []; sep_table = Table(ascii.read('targets/targets_kep/furlan_sample.tsv'))

for t in targets:
	if t in sep_table['KOI']:
		furlan_sep.append(float(np.array(sep_table['Sep'])[np.where(sep_table['KOI'] == t)][0]))
	else:
		furlan_sep.append(np.nan)

#now make the big stellar Kepler properties dictionary
header = ['system', 'kep_teff', 'kep_teff_err', 'kep_rstar', 'kep_rstar_err', 'kep_mstar', 'kep_pri_prcf', 'kep_sec_prcf', 'separation']

star_table = Table()

star_table['system'] = star_target_params['koi']
star_table['kep_teff'] = star_target_params['teff']; star_table['kep_teff_err'] = np.max([star_target_params['teff_err2'], star_target_params['teff_err1']], axis = 0)
star_table['kep_rstar'] = star_target_params['radius']; star_table['kep_rstar_err'] = np.max([star_target_params['radius_err2'], star_target_params['radius_err1']], axis=0)
star_table['kep_mstar'] = star_target_params['mass']

star_table['kep_pri_prcf'] = furlan_primary_prcf
star_table['kep_sec_prcf'] = furlan_secondary_prcf
star_table['separation'] = furlan_sep

#initialize tables for the derived/revised star and planet properties
derived_planet = Table(names = ['pname', 'rp', 'rp_plus', 'rp_minus', 'rs', 'rs_plus', 'rs_minus', 'tp', 'tp_plus', 'tp_minus', 'ts', 'ts_plus', 'ts_minus', 'sp', 'sp_plus', 'sp_minus', 'ss', 'ss_plus', 'ss_minus', 'tstar_index', 'separation'])
derived_star = Table(names = ['sname', 'pteff', 'pteff_plus', 'pteff_minus', 'steff', 'steff_plus', 'steff_minus', 'prad', 'prad_plus', 'prad_minus', 'srad', 'srad_plus', 'srad_minus', 'p_prcf', 'p_prcf_plus', 'p_prcf_minus', 's_prcf', 's_prcf_plus', 's_prcf_minus', 'mp', 'ms', 'q', 'q_plus', 'q_minus'])

for n, dirname in enumerate(star_table['system']):
	try:
		t1, t2, av, r1, ratio, plx = np.genfromtxt('koi'+dirname.zfill(4) + '/samples.txt').T
	except:
		t1, t2, logg1, logg2, av, r1, ratio, plx = np.genfromtxt('koi'+dirname.zfill(4) + '/samples.txt').T

	pri_prcf = np.genfromtxt('koi' + str(dirname).zfill(4) + '/pri_corr.txt')
	sec_prcf = np.genfromtxt('koi' + str(dirname).zfill(4) + '/sec_corr.txt')

	pri_mass = np.genfromtxt('koi' + str(dirname).zfill(4) + '/primary_mass_posterior.txt')
	sec_mass = np.genfromtxt('koi' + str(dirname).zfill(4) + '/secondary_mass_posterior.txt')
	pri_lum = np.genfromtxt('koi' + str(dirname).zfill(4) + '/primary_lum_posterior.txt')
	sec_lum = np.genfromtxt('koi' + str(dirname).zfill(4) + '/secondary_lum_posterior.txt')


	q = sec_mass/pri_mass; q_mean = np.percentile(q, 50); q_plus = np.percentile(q, 84) - q_mean; q_minus = q_mean - np.percentile(q, 16)

	try:
		system_name = dirname.split('_')[0]
	except:
		system_name = dirname

	old_teff_posterior = np.random.normal(star_table['kep_teff'][n], star_table['kep_teff_err'][n], len(pri_prcf))

	pri_temp = t1[np.random.choice(len(t1), size = int(len(pri_prcf)), replace = False)]
	sec_temp = t2[np.random.choice(len(t1), size = int(len(pri_prcf)), replace = False)]
	pt_50 = np.nanpercentile(pri_temp, 50); pt_84 = np.nanpercentile(pri_temp, 84);pt_16 = np.nanpercentile(pri_temp, 16)
	st_50 = np.nanpercentile(sec_temp, 50); st_84 = np.nanpercentile(sec_temp, 84);st_16 = np.nanpercentile(sec_temp, 16)


	pri_rad = r1[np.random.choice(len(t1), size = int(len(pri_prcf)), replace = False)]
	sec_rad = ratio[np.random.choice(len(t1), size = int(len(pri_prcf)), replace = False)]
	pr_50 = np.nanpercentile(pri_rad, 50); pr_84 = np.nanpercentile(pri_rad, 84);pr_16 = np.nanpercentile(pri_rad, 16)
	sr_50 = np.nanpercentile(sec_rad, 50); sr_84 = np.nanpercentile(sec_rad, 84);sr_16 = np.nanpercentile(sec_rad, 16)


	np_prcf = pri_prcf * (pri_rad/np.random.normal(star_table['kep_rstar'][n], star_table['kep_rstar_err'][n], len(pri_prcf)))
	ns_prcf = sec_prcf * (pri_rad/np.random.normal(star_table['kep_rstar'][n], star_table['kep_rstar_err'][n], len(pri_prcf)))
	np_50 = np.nanpercentile(np_prcf, 50); np_84 = np.nanpercentile(np_prcf, 84);np_16 = np.nanpercentile(np_prcf, 16)
	ns_50 = np.nanpercentile(ns_prcf, 50); ns_84 = np.nanpercentile(ns_prcf, 84);ns_16 = np.nanpercentile(ns_prcf, 16)

	derived_star.add_row([system_name, pt_50, pt_84-pt_50, pt_50-pt_16, st_50, st_84-st_50, st_50-st_16, pr_50, pr_84-pr_50, pr_50-pr_16, sr_50, sr_84-sr_50, sr_50-sr_16, np_50, np_84-np_50, np_50-np_16, ns_50, ns_84-ns_50, ns_50-ns_16, np.mean(pri_mass), np.mean(sec_mass), q_mean, q_plus, q_minus])

	with open('star_params.txt', 'a') as f:
		f.write(r'{} & {:.0f}$^{{+{:.0f}}}_{{-{:.0f}}}$ & {:.0f}$^{{+{:.0f}}}_{{-{:.0f}}}$ & {:.0f}$\pm${:.0f} & {:.2f}$^{{+{:.2f}}}_{{-{:.2f}}}$ & {:.2f}$^{{+{:.2f}}}_{{-{:.2f}}}$ & {:.2f} $\pm$ {:.2f} & {:.2f}$^{{+{:.2f}}}_{{-{:.2f}}}$ & {:.2f}$^{{+{:.2f}}}_{{-{:.2f}}}$\\'\
		.format(system_name, pt_50, pt_84-pt_50, pt_50-pt_16, st_50, st_84-st_50, st_50-st_16, star_table['kep_teff'][n], star_table['kep_teff_err'][n], pr_50, pr_84-pr_50, pr_50-pr_16, sr_50, sr_84-sr_50, sr_50-sr_16, star_table['kep_rstar'][n], star_table['kep_rstar_err'][n], np_50, np_84-np_50, np_50-np_16, ns_50, ns_84-ns_50, ns_50-ns_16))
		f.write('\n')
		f.close()

	#calculate the revised planet radius
	#this is simply planet radius correction factor times the original radius
	for k, planet in enumerate(planet_table['pname']):
		if system_name in planet.zfill(7):
			planet_table['tstar_index'][k] = n
			old_planet_radius = np.random.normal(planet_table['radius'][k], planet_table['radius_err'][k], len(pri_prcf))

			new_planet_pradius_posterior = old_planet_radius * pri_prcf * (pri_rad/np.random.normal(star_table['kep_rstar'][n], star_table['kep_rstar_err'][n], len(pri_prcf)))
			new_planet_sradius_posterior = old_planet_radius * sec_prcf * sec_rad * (pri_rad/np.random.normal(star_table['kep_rstar'][n], star_table['kep_rstar_err'][n], len(pri_prcf)))

			new_rp_mean = np.mean(new_planet_pradius_posterior); new_rp_std = np.std(new_planet_pradius_posterior); 
			new_rp_84 = np.abs(np.nanpercentile(np.array(new_planet_pradius_posterior), 84)); new_rp_16 = np.abs(np.nanpercentile(np.array(new_planet_pradius_posterior),16))

			new_rs_mean = np.mean(new_planet_sradius_posterior); new_rs_std = np.std(new_planet_sradius_posterior); 
			new_rs_84 = np.abs(np.nanpercentile(np.array(new_planet_sradius_posterior), 84)); new_rs_16 = np.abs(np.nanpercentile(np.array(new_planet_sradius_posterior),16))

			#calculate the revised teq
			#equation is teq_new = teq_old * (teff_new/teff_old) * sqrt(radius_new/radius_old)
			rad_sample = np.random.normal(star_table['kep_rstar'][n], star_table['kep_rstar_err'][n], len(pri_prcf))
			for p, t in enumerate(rad_sample):
				if t < 0:
					newr = -1
					while newr < 0:
						newr = np.random.normal(star_table['kep_rstar'][n], star_table['kep_rstar_err'][n])
					rad_sample[p] = newr

			teq_pri_new = planet_table['teq'][k] * (pri_temp/old_teff_posterior) * np.sqrt(pri_rad/rad_sample)
			teq_sec_new = planet_table['teq'][k] * (sec_temp/old_teff_posterior) * np.sqrt((pri_rad * sec_rad)/rad_sample)


			new_teqp_mean = np.mean(teq_pri_new); new_teqp_std = np.std(teq_pri_new); 
			new_teqp_84 = np.abs(np.nanpercentile(np.array(teq_pri_new), 84)); new_teqp_16 = np.abs(np.nanpercentile(np.array(teq_pri_new),16))

			new_teqs_mean = np.mean(teq_sec_new); new_teqs_std = np.std(teq_sec_new); 
			new_teqs_84 = np.abs(np.nanpercentile(np.array(teq_sec_new), 84)); new_teqs_16 = np.abs(np.nanpercentile(np.array(teq_sec_new),16))

			#calculate the new sma
			#equation is sma = period**2 * new_mass^(1/3)
			period_draw = np.random.normal(planet_table['period'][k], planet_table['period_err'][k], len(pri_prcf))

			new_pri_sma = (period_draw**2 * pri_mass)**(1/3)
			new_sec_sma = (period_draw**2 * sec_mass)**(1/3)

			new_smap_mean = np.mean(new_pri_sma); new_smap_std = np.std(new_pri_sma); 
			new_smap_84 = np.abs(np.nanpercentile(np.array(new_pri_sma), 84)); new_smap_16 = np.abs(np.nanpercentile(np.array(new_pri_sma),16))

			new_smas_mean = np.mean(new_sec_sma); new_smas_std = np.std(new_sec_sma); 
			new_smas_84 = np.abs(np.nanpercentile(np.array(new_sec_sma), 84)); new_smas_16 = np.abs(np.nanpercentile(np.array(new_sec_sma),16))


			#calculate new instellation
			#equation is S = Lum_new / sma_new^2

			new_ps = pri_lum/new_pri_sma**2; new_ss = sec_lum/new_sec_sma**2

			new_smap_mean = np.mean(new_ps); new_smap_std = np.std(new_ps); 
			new_smap_84 = np.abs(np.nanpercentile(np.array(new_ps), 84)); new_smap_16 = np.abs(np.nanpercentile(np.array(new_ps),16))

			new_smas_mean = np.mean(new_ss); new_smas_std = np.std(new_ss); 
			new_smas_84 = np.abs(np.nanpercentile(np.array(new_ss), 84)); new_smas_16 = np.abs(np.nanpercentile(np.array(new_ss),16))

			derived_planet.add_row([planet, new_rp_mean, new_rp_84-new_rp_mean, new_rp_mean-new_rp_16, new_rs_mean, new_rs_84-new_rs_mean, new_rs_mean-new_rs_16,\
					new_teqp_mean, new_teqp_84-new_teqp_mean, new_teqp_mean-new_teqp_16, new_teqs_mean, new_teqs_84-new_teqs_mean, new_teqs_mean-new_teqs_16,\
					new_smap_mean, new_smap_84-new_smap_mean, new_smap_mean-new_smap_16, new_smas_mean, new_smas_84-new_smas_mean, new_smas_mean-new_smas_16, n, star_table['separation'][n]])

			with open('revised_radii_teq.txt', 'a') as f:
				f.write(r'{} & {:.2f}$^{{+{:.2f}}}_{{-{:.2f}}}$ & {:.2f}$^{{+{:.2f}}}_{{-{:.2f}}}$ & {:.2f}$\pm${:.2f} & {:.0f}$^{{+{:.0f}}}_{{-{:.0f}}}$ & {:.0f}$^{{+{:.0f}}}_{{-{:.0f}}}$ & {:.0f} & {:.2f}$^{{+{:.2f}}}_{{-{:.2f}}}$ & {:.2f}$^{{+{:.2f}}}_{{-{:.2f}}}$ & {:.2f}$\pm${:.2f}\\'\
				.format(planet, new_rp_mean, new_rp_84-new_rp_mean, new_rp_mean-new_rp_16, new_rs_mean, new_rs_84-new_rs_mean, new_rs_mean-new_rs_16, planet_table['radius'][k], planet_table['radius_err'][k],\
					new_teqp_mean, new_teqp_84-new_teqp_mean, new_teqp_mean-new_teqp_16, new_teqs_mean, new_teqs_84-new_teqs_mean, new_teqs_mean-new_teqs_16, planet_table['teq'][k],\
					new_smap_mean, new_smap_84-new_smap_mean, new_smap_mean-new_smap_16, new_smas_mean, new_smas_84-new_smas_mean, new_smas_mean-new_smas_16, planet_table['s'][k], planet_table['s_err'][k]))
				f.write('\n')
				f.close()

print('total number of stars analyzed: ', len(derived_star), ', total number of planets: ', len(derived_planet))

pprcf = np.nanpercentile(derived_star['p_prcf'], 50)
pprcf_84 = np.nanpercentile(derived_star['p_prcf'], 84)
pprcf_16 = np.nanpercentile(derived_star['p_prcf'], 16)

sprcf = np.nanpercentile(derived_star['s_prcf'], 50)
sprcf_84 = np.nanpercentile(derived_star['s_prcf'], 84)
sprcf_16 = np.nanpercentile(derived_star['s_prcf'], 16)

print("average primary prcf: {:.3f} + {:.3f} - {:.3f}".format(pprcf, pprcf_84 - pprcf, pprcf - pprcf_16))
print('average secondary prcf: {:.3f} + {:.3f} - {:.3f}'.format(sprcf, sprcf_84 - sprcf, sprcf - sprcf_16))

pm = np.nanpercentile((derived_star['mp'] - star_table['kep_mstar'])/star_table['kep_mstar'], 50)
pm_84 = np.nanpercentile((derived_star['mp'] - star_table['kep_mstar'])/star_table['kep_mstar'], 84)
pm_16 = np.nanpercentile((derived_star['mp'] - star_table['kep_mstar'])/star_table['kep_mstar'], 16)

sm = np.nanpercentile((star_table['kep_mstar'] - derived_star['ms'])/star_table['kep_mstar'], 50)
sm_84 = np.nanpercentile((star_table['kep_mstar'] - derived_star['ms'])/star_table['kep_mstar'], 84)
sm_16 = np.nanpercentile((star_table['kep_mstar'] - derived_star['ms'])/star_table['kep_mstar'], 16)

print('primary star fraction mass change: {:.3f} + {:.3f} - {:.3f}'.format(pm, pm_84 - pm, pm - pm_16))
print('secondary star fraction mass change: {:.3f} + {:.3f} - {:.3f}'.format(sm, sm_84 - sm, sm - sm_16))

##### TEFFDIFF code ######

pteff_mean = np.nanpercentile(derived_star['pteff'] - star_table['kep_teff'], 50)
pteff_84 = np.nanpercentile(derived_star['pteff'] - star_table['kep_teff'], 84)
pteff_16 = np.nanpercentile(derived_star['pteff'] - star_table['kep_teff'], 16)
steff_mean = np.nanpercentile(star_table['kep_teff'] - derived_star['steff'], 50)
steff_84 = np.nanpercentile(star_table['kep_teff'] - derived_star['steff'], 84)
steff_16 = np.nanpercentile(star_table['kep_teff'] - derived_star['steff'], 16)

print('avg primary star teff change (K): {:.3f} + {:.3f} - {:.3f}'.format(pteff_mean, pteff_84 - pteff_mean, pteff_mean - pteff_16))
print('avg secondary star teff change (K): {:.3f} + {:.3f} - {:.3f}'.format(steff_mean, pteff_84 - steff_mean, steff_mean - pteff_16))

prad_mean = np.nanpercentile((derived_planet['rp'] - planet_table['radius'])/planet_table['radius'], 50)
prad_84 = np.nanpercentile((derived_planet['rp'] - planet_table['radius'])/planet_table['radius'], 84)
prad_16 = np.nanpercentile((derived_planet['rp'] - planet_table['radius'])/planet_table['radius'], 16)
srad_mean = np.nanpercentile((planet_table['radius'] - derived_planet['rs'])/planet_table['radius'], 50)
srad_84 = np.nanpercentile((planet_table['radius'] - derived_planet['rs'])/planet_table['radius'], 84)
srad_16 = np.nanpercentile((planet_table['radius'] - derived_planet['rs'])/planet_table['radius'], 16)

print('avg primary host planet rad change (fraction): {:.3f} + {:.3f} - {:.3f}'.format(prad_mean, prad_84 - prad_mean, prad_mean - prad_16))
print('avg secondary host planet rad change (fraction): {:.3f} + {:.3f} - {:.3f}'.format(srad_mean, srad_84 - srad_mean, srad_mean - srad_16))

ps_mean = np.nanpercentile((derived_planet['sp'] - planet_table['s'])/planet_table['s'], 50)
ps_84 = np.nanpercentile((derived_planet['sp'] - planet_table['s'])/planet_table['s'], 84)
ps_16 = np.nanpercentile((derived_planet['sp'] - planet_table['s'])/planet_table['s'], 16)
ss_mean = np.nanpercentile((planet_table['s'] - derived_planet['ss'])/planet_table['s'], 50)
ss_84 = np.nanpercentile((planet_table['s'] - derived_planet['ss'])/planet_table['s'], 84)
ss_16 = np.nanpercentile((planet_table['s'] - derived_planet['ss'])/planet_table['s'], 16)

print('avg primary host planet instellation change (fraction): {:.3f} + {:.3f} - {:.3f}'.format(ps_mean, ps_84 - ps_mean, ps_mean - prad_16))
print('avg secondary host planet instellation change (fraction): {:.3f} + {:.3f} - {:.3f}'.format(ss_mean, ss_84 - ss_mean, ss_mean - srad_16))


### NOW PLOT EVERYTHING ###
plt.figure()
plt.scatter(derived_star['p_prcf']/star_table['kep_pri_prcf'], derived_star['s_prcf']/star_table['kep_sec_prcf'], marker = '.', s = 100, color = 'k')
plt.axvline(1, label = 'Agreement', linestyle = '--', color = '0.5', zorder = 0)
plt.axhline(1, linestyle = '--', color = '0.5', zorder = 0)
plt.xlabel('Primary PRCF ratio (this work/Furlan+2017)', fontsize = 13)
plt.ylabel('Secondary PRCF ratio (this work/Furlan+2017)', fontsize = 13)
plt.minorticks_on()
plt.legend(loc = 'best', fontsize = 12)
plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
plt.tick_params(bottom=True, top =True, left=True, right=True)
plt.tick_params(which='both', labelsize = 14, direction='in')
plt.tick_params('both', length=8, width=1.5, which='major')
plt.tick_params('both', length=4, width=1, which='minor')
plt.tight_layout()
plt.savefig('prcf_compare.pdf')
plt.close()

fix, ax = plt.subplots()
ax.scatter(star_table['kep_teff'], derived_star['pteff'], marker = '.', s = 100, color = 'darkblue', label = 'Primary', zorder = 1)
ax.scatter(star_table['kep_teff'], derived_star['steff'], marker = '.', s = 100, color = 'darkorange', label = 'Secondary', zorder = 1)
ax.errorbar(star_table['kep_teff'], derived_star['pteff'], xerr = star_table['kep_teff_err'], yerr = [derived_star['pteff_plus'], derived_star['pteff_minus']], linestyle = 'None', color = 'darkblue', zorder = 0, capsize = 3, elinewidth = 1)
ax.errorbar(star_table['kep_teff'], derived_star['steff'], xerr = star_table['kep_teff_err'], yerr = [derived_star['steff_plus'], derived_star['steff_minus']], linestyle = 'None', color = 'darkorange', zorder = 0, capsize = 3, elinewidth = 1)
for n, t in enumerate(derived_star['pteff']):
	ax.plot([star_table['kep_teff'][n], star_table['kep_teff'][n]], [derived_star['steff'][n], t], color = 'k', linewidth = 1, zorder = 0.5)
ax.plot([min(star_table['kep_teff']) - 800, max(star_table['kep_teff']) + 800], [min(star_table['kep_teff']) - 800, max(star_table['kep_teff']) + 800] , label = '1:1', linestyle = ':', color = 'k')
plt.minorticks_on()
ax.set_xlim(2950, 7200)
ax.set_ylim(2950, 7200)
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = 14, direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.set_xlabel(r'$T_{eff}$ (Kepler; K)', fontsize = 14)
ax.set_ylabel('{}'.format(r'Fitted $T_{eff}$ (this work; K)'), fontsize = 14)
ax.legend(loc = 'best', fontsize = 12)
plt.tight_layout()
plt.savefig('teff_diff.pdf')
plt.close()

bins = np.linspace(float(min(derived_star['steff'])), float(max(derived_star['pteff'])), 15)
fig, [ax1, ax2, ax3] = plt.subplots(nrows = 3, gridspec_kw = dict(hspace = 0), sharex = True, sharey = True, figsize = (7,6))
ax1.hist(star_table['kep_teff'], color = 'k', linewidth = 2, histtype = 'step', label = r'Kepler $T_{eff}$', bins = bins)
ax2.hist(derived_star['pteff'], color = 'darkblue', linewidth = 2, histtype = 'step', hatch = '/', label = r'Primary $T_{eff}$', bins = bins)
ax3.hist(derived_star['steff'], color = 'darkorange', linewidth = 2, histtype = 'step', hatch = 'x', label = r'Secondary $T_{eff}$', bins = bins)

ax1.legend(loc = 'best', fontsize = 12)
ax2.legend(loc = 'best', fontsize = 12)
ax3.legend(loc = 'best', fontsize = 12)

plt.minorticks_on()

ax1.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax1.tick_params(bottom=True, top =True, left=True, right=True)
ax1.tick_params(which='both', labelsize = 14, direction='in')
ax1.tick_params('both', length=8, width=1.5, which='major')
ax1.tick_params('both', length=4, width=1, which='minor')

ax2.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax2.tick_params(bottom=True, top =True, left=True, right=True)
ax2.tick_params(which='both', labelsize = 14, direction='in')
ax2.tick_params('both', length=8, width=1.5, which='major')
ax2.tick_params('both', length=4, width=1, which='minor')

ax3.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax3.tick_params(bottom=True, top =True, left=True, right=True)
ax3.tick_params(which='both', labelsize = 14, direction='in')
ax3.tick_params('both', length=8, width=1.5, which='major')
ax3.tick_params('both', length=4, width=1, which='minor')

ax3.set_xlabel('Temperature (K)', fontsize = 14)
ax2.set_ylabel('N', fontsize = 14)
plt.tight_layout()
plt.savefig('teff_hist.pdf')
plt.close()

# bins = np.arange(np.log10(0.3), np.log10(7.1), 0.06)
bins = np.arange(0.35, 6, 0.2)

fig, [ax1, ax2, ax3] = plt.subplots(nrows = 3, gridspec_kw = dict(hspace = 0), sharex = True, figsize = (7,6))
a = ax1.hist(planet_table['radius'], color = 'k', linewidth = 2, histtype = 'step', label = r'Kepler $R_{P}$', bins = bins)
b = ax2.hist(derived_planet['rp'], color = 'darkblue', linewidth = 2, histtype = 'step', hatch = '/', label = r'Primary host $R_{p}$', bins = bins)
c = ax3.hist(derived_planet['rs'], color = 'darkorange', linewidth = 2, histtype = 'step', hatch = 'x', label = r'Secondary host $R_{p}$', bins = bins)
d = ax1.axvline(1.9, linewidth = 2, color = 'k', label = r'1.9 $R_{\oplus}$')
ax2.axvline(1.9, linewidth = 2, color = 'k')
ax3.axvline(1.9, linewidth = 2, color = 'k')

ax1.legend(loc = 'best', fontsize = 12, framealpha = 0)
ax2.legend(loc = 'best', fontsize = 12, framealpha = 0)
ax3.legend(loc = 'best', fontsize = 12, framealpha = 0)

ax1.minorticks_on()
ax1.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax1.tick_params(bottom=True, top =True, left=True, right=True)
ax1.tick_params(which='both', labelsize = 14, direction='in')
ax1.tick_params('both', length=8, width=1.5, which='major')
ax1.tick_params('both', length=4, width=1, which='minor')

ax2.minorticks_on()
ax2.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax2.tick_params(bottom=True, top =True, left=True, right=True)
ax2.tick_params(which='both', labelsize = 14, direction='in')
ax2.tick_params('both', length=8, width=1.5, which='major')
ax2.tick_params('both', length=4, width=1, which='minor')

ax3.minorticks_on()
ax3.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax3.tick_params(bottom=True, top =True, left=True, right=True)
ax3.tick_params(which='both', labelsize = 14, direction='in')
ax3.tick_params('both', length=8, width=1.5, which='major')
ax3.tick_params('both', length=4, width=1, which='minor')

# ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(round(10**y, 2))))
ax3.set_xlabel(r'$R_{P} (R_{\oplus})$', fontsize = 14)
ax2.set_ylabel('N', fontsize = 14)
ax1.set_rasterization_zorder(0)
ax2.set_rasterization_zorder(0)
# ax3.set_xscale('log')

plt.tight_layout()
plt.savefig('radius_hist.pdf')
plt.close()

##### PRIMARY PLOTS #####
fix, ax = plt.subplots()
ax.scatter(planet_table['radius'], derived_planet['rp'], marker = '.', s = 100, color = 'darkblue', label = 'Primary host', zorder = 1)
ax.errorbar(planet_table['radius'], derived_planet['rp'], xerr = planet_table['radius_err'], yerr = [derived_planet['rp_minus'], derived_planet['rp_plus']], linestyle = 'None', color = 'darkblue', zorder = 1, alpha = 0.5, elinewidth = 1)
ax.axhline(1, label = r'1 R$_{\bigoplus}$', linestyle = '--', color = '0.8', linewidth = 2, zorder = 0)
ax.axhline(1.8, label = r'1.8 R$_{\bigoplus}$', linestyle = '-.', color = '0.5', linewidth = 2, zorder = 0)
ax.axvline(1, linestyle = '--', color = '0.8', linewidth = 2, zorder = 0)
ax.axvline(1.8, linestyle = '-.', color = '0.5', linewidth = 2, zorder = 0)
ax.plot([0.3,3], [0.3,3], label = '1:1', linestyle = ':', linewidth = 1.2, color = '0.3', zorder = 0)
ax.set_xlim(0.3, 3)
# ax.set_ylim(0.3, 2.5)
ax.set_yscale('log')
plt.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = 14, direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.set_xlabel(r'R$_{p}$ (Kepler; R$_{\bigoplus}$)', fontsize = 14)
ax.set_ylabel(r'R$_{p}$ (this work; R$_{\bigoplus}$)', fontsize = 14)
ax.legend(loc = 'best', fontsize = 12)
plt.tight_layout()
plt.savefig('rp_diff_primary.pdf')
plt.close()

fix, ax = plt.subplots()
ax.scatter(derived_planet['sp'], derived_planet['rp'], marker = '.', s = 100, color = 'darkblue', label = 'Primary host', zorder = 1)
ax.scatter(planet_table['s'], planet_table['radius'], marker = '.', s = 100, edgecolor = 'darkblue', facecolors = 'None', label = 'Kepler', zorder = 1)
ax.errorbar(derived_planet['sp'], derived_planet['rp'], xerr = [derived_planet['sp_minus'], derived_planet['sp_plus']], yerr = [derived_planet['rp_minus'], derived_planet['rp_plus']], linestyle = 'None', color = 'darkblue', zorder = 1, alpha = 0.3, elinewidth = 1)
ax.errorbar(planet_table['s'], planet_table['radius'], xerr = planet_table['s_err'], yerr = planet_table['radius_err'], linestyle = 'None', color = 'darkblue', zorder = 1, alpha = 0.3, elinewidth = 1)
for n in range(len(derived_planet['rp'])):
	ax.annotate("", xy=(derived_planet['sp'][n], derived_planet['rp'][n]), xytext=(planet_table['s'][n], planet_table['radius'][n]), arrowprops=dict(arrowstyle="->"))
ax.axvspan(3e2, runaway_greenhouse(5870), alpha = 0.1, color = 'xkcd:bright red', zorder = 0)
ax.axvspan(3e2, recent_venus(5870), alpha = 0.1, color = 'xkcd:scarlet', zorder = 0)
# ax.text(1, 0.5, 'Cons. HZ', fontsize = 10)
ax.axvspan(0, max_greenhouse_limit(5870), alpha = 0.1, color = 'xkcd:azure', zorder = 0)
ax.axhline(1, label = r'1 R$_{\bigoplus}$', linestyle = '--', color = '0.8', linewidth = 2, zorder = 0)
ax.axhline(1.8, label = r'1.8 R$_{\bigoplus}$', linestyle = '-.', color = '0.5', linewidth = 2, zorder = 0)
plt.minorticks_on()
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_xlim(3e2, 8e-2)
plt.gca().invert_xaxis()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = 14, direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.set_xlabel(r'Instellation (S$_{\bigoplus}$)', fontsize = 14)
ax.set_ylabel(r'R$_{P} (R_{\bigoplus}$)', fontsize = 14)
ax.legend(loc = 'upper left', fontsize = 12)
plt.tight_layout()
plt.savefig('s_vs_r_primary.pdf')
plt.close()

kde = KernelDensity(kernel='gaussian', bandwidth = 0.2).fit(np.array([np.log10(planet_table['period']*365.25), derived_planet['rp']]).T)

X, Y = np.meshgrid(np.arange(np.log10(1e-2), max(np.log10(planet_table['period']*365.25))+1.5, 0.05), np.arange(min(derived_planet['rp'])-0.5, max(derived_planet['rp'])+1.5, 0.05))
xy = np.vstack([X.ravel(), Y.ravel()]).T
Z = np.exp(kde.score_samples(xy))
Z = Z.reshape(X.shape)
levels = np.linspace(0, Z.max(), 40)

def line(x):
	return 1.74*(x/10)**-0.13

periods = np.linspace(-2, 4, 100)
periods_days = (10**periods)

fix, ax = plt.subplots()
ax.scatter(np.log10(planet_table['period']*365.25), derived_planet['rp'], marker = '.', s = 100, color = 'darkblue', label = 'Primary host', zorder = 1)
# ax.scatter(planet_table['period']*365.25, planet_table['radius'], marker = '.', s = 100, edgecolor = 'darkblue', facecolors = 'None', label = 'Kepler', zorder = 1)
ax.errorbar(np.log10(planet_table['period']*365.25), derived_planet['rp'], xerr = planet_table['period_err'], yerr = [derived_planet['rp_minus'], derived_planet['rp_plus']], linestyle = 'None', color = 'darkblue', zorder = 1, alpha = 0.3, elinewidth = 1)
# ax.errorbar(planet_table['period']*365.25, planet_table['radius'], xerr = planet_table['period_err'], yerr = planet_table['radius_err'], linestyle = 'None', color = 'darkblue', zorder = 1, alpha = 0.3, elinewidth = 1)
# for n in range(len(derived_planet['rp'])):
	# ax.annotate("", xy=(planet_table['period'][n]*365.25, derived_planet['rp'][n]), xytext=(planet_table['period'][n]*365.25, planet_table['radius'][n]), arrowprops=dict(arrowstyle="->"))
plt.contourf(X, Y, Z, cmap=plt.cm.Blues, levels=levels, zorder = -99)
plt.plot(periods, line(periods_days), color = 'k', linewidth = 2, linestyle = '--', label = 'Petigura+2022 (single stars)')
# ax.axhline(1, label = r'1 R$_{\bigoplus}$', linestyle = '--', color = '0.8', linewidth = 2, zorder = 0))
# ax.axhline(1.8, label = r'1.8 R$_{\bigoplus}$', linestyle = '-.', color = '0.5', linewidth = 2, zorder = 0))
plt.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = 14, direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.set_xlabel(r'log$_{10}$(Period) (days)', fontsize = 14)
ax.set_ylabel(r'log$_{10}$(R$_{P}) (R_{\bigoplus}$)', fontsize = 14)
# ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(min(derived_planet['rp']) - 0.05, max(derived_planet['rp']) + 0.5)
ax.set_xlim(min(np.log10(planet_table['period'] * 365.25)) - 1, max(np.log10(planet_table['period'] * 365.25)) + 1)
ax.legend(loc = 'best', fontsize = 12)
plt.tight_layout()
plt.savefig('p_vs_r_primary.pdf')
plt.close()

def line(x, m, b):
	return m*x + b

popt, pcov = curve_fit(line, derived_planet['separation'][~np.isnan(derived_planet['separation'])], derived_planet['rp'][~np.isnan(derived_planet['separation'])], sigma = [max(a, b) for a,b in zip(derived_planet['rp_plus'][~np.isnan(derived_planet['separation'])], derived_planet['rp_minus'][~np.isnan(derived_planet['separation'])])])
x_arr = np.arange(np.nanmin(derived_planet['separation']), np.nanmax(derived_planet['separation']), 0.02)

fix, ax = plt.subplots()
ax.scatter(derived_planet['separation'], derived_planet['rp'], marker = '.', s = 100, color = 'darkblue', label = 'Primary host', zorder = 1)
ax.errorbar(derived_planet['separation'], derived_planet['rp'], yerr = [derived_planet['rp_minus'], derived_planet['rp_plus']], linestyle = 'None', color = 'darkblue', zorder = 1, alpha = 0.3, elinewidth = 1)
ax.plot(x_arr, line(x_arr, *popt), linewidth = 2, color = 'k', label = 'best-fit line')
ax.axhline(1, label = r'1 R$_{\bigoplus}$', linestyle = '--', color = '0.8', linewidth = 2, zorder = 0)
ax.axhline(1.8, label = r'1.8 R$_{\bigoplus}$', linestyle = '-.', color = '0.5', linewidth = 2, zorder = 0)
plt.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = 14, direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.set_xlabel(r'$\rho$ (arcsec)', fontsize = 14)
ax.set_ylabel(r'R$_{P} (R_{\bigoplus}$)', fontsize = 14)
# ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(min(derived_planet['rp']) - 0.05, max(derived_planet['rp']) + 0.5)
ax.legend(loc = 'upper left', fontsize = 12)
plt.tight_layout()
plt.savefig('rho_vs_r_primary.pdf')
plt.close()

fix, ax = plt.subplots()
ax.scatter(derived_planet['sp'], [derived_star['pteff'][int(k)] for k in planet_table['tstar_index']], marker = '.', s = 100, color = 'darkblue', label = 'Primary host', zorder = 1)
ax.scatter(planet_table['s'], [star_table['kep_teff'][int(k)] for k in planet_table['tstar_index']], marker = '.', s = 100,  edgecolor = 'darkblue', facecolors = 'None', label = 'Kepler', zorder = 1)
ax.errorbar(derived_planet['sp'], [derived_star['pteff'][int(k)] for k in planet_table['tstar_index']], xerr = [derived_planet['sp_minus'], derived_planet['sp_plus']], yerr = [[derived_star['pteff_minus'][int(k)] for k in planet_table['tstar_index']], [derived_star['pteff_plus'][int(k)] for k in planet_table['tstar_index']]], linestyle = 'None', color = 'darkblue', zorder = 1, alpha = 0.5, elinewidth = 1)
ax.errorbar(planet_table['s'], [star_table['kep_teff'][int(k)] for k in planet_table['tstar_index']], xerr = planet_table['s_err'], yerr = [star_table['kep_teff_err'][int(k)] for k in planet_table['tstar_index']], linestyle = 'None', color = 'darkblue', zorder = 1, alpha = 0.5, elinewidth = 1)
for n in range(len([derived_star['pteff'][int(k)] for k in planet_table['tstar_index']])):
	ax.annotate("", xy=(derived_planet['sp'][n], [derived_star['pteff'][int(k)] for k in planet_table['tstar_index']][n]), xytext=(planet_table['s'][n], [star_table['kep_teff'][int(k)] for k in planet_table['tstar_index']][n]), arrowprops=dict(arrowstyle="->"))
plt.plot(recent_venus(tstars), tstars, linestyle = ':', color = 'k', linewidth = 2, label= 'Optimistic inner HZ')
plt.plot(max_greenhouse_limit(tstars), tstars, linestyle = '--', color = 'k', linewidth = 2, label = 'Outer HZ')
plt.plot(runaway_greenhouse(tstars), tstars, linestyle = '-.', color = 'k', linewidth = 2, label = 'Cons. inner HZ')
plt.gca().invert_xaxis()
ax.set_xscale('log')
plt.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = 14, direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.set_xlabel(r'Instellation (S$_{\bigoplus}$)', fontsize = 14)
ax.set_ylabel(r'T$_{\star}$ (K)', fontsize = 14)
ax.legend(loc = 'upper left', fontsize = 12, ncol = 2)
plt.tight_layout()
plt.savefig('s_vs_t_primary.pdf')
plt.close()

fix, ax = plt.subplots()
ax.scatter([derived_star['q'][int(k)] for k in planet_table['tstar_index']], derived_planet['rp'], marker = '.', s = 100, color = 'darkblue', label = 'Primary host', zorder = 1)
ax.errorbar([derived_star['q'][int(k)] for k in planet_table['tstar_index']], derived_planet['rp'], xerr = [[derived_star['q_minus'][int(k)] for k in planet_table['tstar_index']], [derived_star['q_plus'][int(k)] for k in planet_table['tstar_index']]], yerr = [derived_planet['rp_minus'], derived_planet['rp_plus']], linestyle = 'None', color = 'darkblue', zorder = 1, alpha = 0.5, elinewidth = 1)
plt.minorticks_on()
ax.set_yscale('log')
ax.set_xlim(0.25, 1)
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = 14, direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.set_xlabel(r'Binary mass ratio q', fontsize = 14)
ax.set_ylabel(r'$R_{p} (R_{\oplus})$', fontsize = 14)
ax.legend(loc = 'best', fontsize = 12, ncol = 2)
plt.tight_layout()
plt.savefig('q_vs_r_primary.pdf')
plt.close()


#### SECONDARY PLOTS #####

fix, ax = plt.subplots()
ax.scatter(planet_table['radius'], derived_planet['rs'], marker = '.', s = 100, color = 'darkorange', label = 'Secondary host', zorder = 1)
ax.errorbar(planet_table['radius'], derived_planet['rs'], xerr = planet_table['radius_err'], yerr = [derived_planet['rs_minus'], derived_planet['rs_plus']], linestyle = 'None', color = 'darkorange', zorder = 1, alpha = 0.5, elinewidth = 1)
ax.axhline(1, label = r'1 R$_{\bigoplus}$', linestyle = '--', color = '0.8', linewidth = 2, zorder = 0)
ax.axhline(1.8, label = r'1.8 R$_{\bigoplus}$', linestyle = '-.', color = '0.5', linewidth = 2, zorder = 0)
ax.axvline(1, linestyle = '--', color = '0.8', linewidth = 2, zorder = 0)
ax.axvline(1.8, linestyle = '-.', color = '0.5', linewidth = 2, zorder = 0)
ax.plot([0.3,3], [0.3,3], label = '1:1', linestyle = ':', linewidth = 1.2, color = '0.3', zorder = 0)
ax.set_xlim(0.3, 3)
# ax.set_ylim(0.3, 7.5)
ax.set_yscale('log')
plt.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = 14, direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.set_xlabel(r'R$_{p}$ (Kepler; R$_{\bigoplus}$)', fontsize = 14)
ax.set_ylabel(r'R$_{p}$ (this work; R$_{\bigoplus}$)', fontsize = 14)
ax.legend(loc = 'best', fontsize = 12)
plt.tight_layout()
plt.savefig('rp_diff_secondary.pdf')
plt.close()

fix, ax = plt.subplots()
ax.scatter(derived_planet['ss'], derived_planet['rs'], marker = '.', s = 100, color = 'darkorange', label = 'Secondary host', zorder = 1)
ax.scatter(planet_table['s'], planet_table['radius'], marker = '.', s = 100, edgecolor = 'darkorange', facecolors = 'None', label = 'Kepler', zorder = 1)
ax.errorbar(derived_planet['ss'], derived_planet['rs'], xerr = [derived_planet['ss_minus'], derived_planet['ss_plus']], yerr = [derived_planet['rs_minus'], derived_planet['rs_plus']], linestyle = 'None', color = 'darkorange', zorder = 1, alpha = 0.3, elinewidth = 1)
ax.errorbar(planet_table['s'], planet_table['radius'], xerr = planet_table['s_err'], yerr = planet_table['radius_err'], linestyle = 'None', color = 'darkorange', zorder = 1, alpha = 0.3, elinewidth = 1)
for n in range(len(derived_planet['rs'])):
	ax.annotate("", xy=(derived_planet['ss'][n], derived_planet['rs'][n]), xytext=(planet_table['s'][n], planet_table['radius'][n]), arrowprops=dict(arrowstyle="->"))
ax.axvspan(3e2, runaway_greenhouse(5870), alpha = 0.1, color = 'xkcd:bright red', zorder = 0)
ax.axvspan(3e2, recent_venus(5870), alpha = 0.1, color = 'xkcd:scarlet', zorder = 0)
# ax.text(1, 0.5, 'Cons. HZ', fontsize = 10)
ax.axvspan(0, max_greenhouse_limit(5870), alpha = 0.1, color = 'xkcd:azure', zorder = 0)
ax.axhline(1, label = r'1 R$_{\bigoplus}$', linestyle = '--', color = '0.8', linewidth = 2, zorder = 0)
ax.axhline(1.8, label = r'1.8 R$_{\bigoplus}$', linestyle = '-.', color = '0.5', linewidth = 2, zorder = 0)
plt.minorticks_on()
ax.set_xscale('log')
ax.set_xlim(3e2, 8e-2)
ax.set_yscale('log')
# plt.gca().invert_xaxis()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = 14, direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.set_xlabel(r'Instellation (S$_{\bigoplus}$)', fontsize = 14)
ax.set_ylabel(r'R$_{P} (R_{\bigoplus}$)', fontsize = 14)
ax.legend(loc = 'upper left', fontsize = 12)
plt.tight_layout()
plt.savefig('s_vs_r_secondary.pdf')
plt.close()

fix, ax = plt.subplots()
ax.scatter(derived_planet['ss'], [derived_star['steff'][int(k)] for k in planet_table['tstar_index']], marker = '.', s = 100, color = 'darkorange', label = 'Secondary host', zorder = 1)
ax.scatter(planet_table['s'], [star_table['kep_teff'][int(k)] for k in planet_table['tstar_index']], marker = '.', s = 100,  edgecolor = 'darkorange', facecolors = 'None', label = 'Kepler', zorder = 1)
ax.errorbar(derived_planet['ss'], [derived_star['steff'][int(k)] for k in planet_table['tstar_index']], xerr = [derived_planet['ss_minus'], derived_planet['ss_plus']], yerr = [[derived_star['steff_minus'][int(k)] for k in planet_table['tstar_index']], [derived_star['pteff_plus'][int(k)] for k in planet_table['tstar_index']]], linestyle = 'None', color = 'darkorange', zorder = 1, alpha = 0.5, elinewidth = 1)
ax.errorbar(planet_table['s'], [star_table['kep_teff'][int(k)] for k in planet_table['tstar_index']], xerr = planet_table['s_err'], yerr = [star_table['kep_teff_err'][int(k)] for k in planet_table['tstar_index']], linestyle = 'None', color = 'darkorange', zorder = 1, alpha = 0.5, elinewidth = 1)
for n in range(len([derived_star['steff'][int(k)] for k in planet_table['tstar_index']])):
	ax.annotate("", xy=(derived_planet['ss'][n], [derived_star['steff'][int(k)] for k in planet_table['tstar_index']][n]), xytext=(planet_table['s'][n], [star_table['kep_teff'][int(k)] for k in planet_table['tstar_index']][n]), arrowprops=dict(arrowstyle="->"))
plt.plot(recent_venus(tstars), tstars, linestyle = ':', color = 'k', linewidth = 2, label= 'Optimistic inner HZ')
plt.plot(max_greenhouse_limit(tstars), tstars, linestyle = '--', color = 'k', linewidth = 2, label = 'Outer HZ')
plt.plot(runaway_greenhouse(tstars), tstars, linestyle = '-.', color = 'k', linewidth = 2, label = 'Cons. inner HZ')
plt.gca().invert_xaxis()
ax.set_xscale('log')
ax.set_yscale('log')
plt.minorticks_on()
ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
ax.tick_params(bottom=True, top =True, left=True, right=True)
ax.tick_params(which='both', labelsize = 14, direction='in')
ax.tick_params('both', length=8, width=1.5, which='major')
ax.tick_params('both', length=4, width=1, which='minor')
ax.set_xlabel(r'Instellation (S$_{\bigoplus}$)', fontsize = 14)
ax.set_ylabel(r'T$_{\star}$ (K)', fontsize = 14)
ax.legend(loc = 'upper left', fontsize = 12, ncol = 2)
plt.tight_layout()
plt.savefig('s_vs_t_secondary.pdf')
plt.close()