import batman
import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(2454966.6983 - 1, 2454966.6983 + 1, 1000) # time array in days (centered on transit)

params = batman.TransitParams()       # object to hold parameters
params.t0 = 2454966.6983              # time of inferior conjunction
params.per = 289.863876               # orbital period in days
params.rp = 0.0275                    # planet radius (in units of stellar radii)
params.a = 94.4                       # semi-major axis (in units of stellar radii)
params.inc = 89.76                    # orbital inclination (in degrees)
params.ecc = 0                        # eccentricity
params.w = 90                         # longitude of periastron (in degrees)
params.limb_dark = "quadratic"        # limb darkening model
params.u = [0.1, 0.3]                 # limb darkening coefficients

t = np.linspace(-0.025, 0.025, 1000)  #times at which to calculate light curve
m = batman.TransitModel(params, t)    #initializes model
flux = m.light_curve(params)           # calculates light curve

fig=plt.figure(figsize=(10, 6))
plt.plot(t, flux, color="blue", label="Model Light Curve")
plt.xlabel("Time from transit center (days)")
plt.ylabel("Relative flux")
plt.title("Exoplanet Transit Light Curve")
plt.legend()
plt.grid(True)
plt.show()
fig.savefig('/root/2_veneti_e_mezzo/src/daneel/detection/Kepler-22_b_assignment1_taskF.png', dpi=fig.dpi)