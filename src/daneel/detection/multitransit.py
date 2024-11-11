import batman
import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(2454966.6983 - 1, 2454966.6983 + 1, 1000) 

params = batman.TransitParams()       # object to hold parameters for first planet
params.t0 = 2454966.6983              # time of inferior conjunction
params.per = 289.863876               # orbital period in days
params.rp = 0.0275                    # planet radius (in units of stellar radii)
params.a = 94.4                       # semi-major axis (in units of stellar radii)
params.inc = 89.76                    # orbital inclination (in degrees)
params.ecc = 0                        # eccentricity
params.w = 90                         # longitude of periastron (in degrees)
params.limb_dark = "quadratic"        # limb darkening model
params.u = [0.630, 0.076]             # limb darkening coefficients

m = batman.TransitModel(params, time)

flux = m.light_curve(params)

#Task2B - setting the radius of the second planet to 0.5*R
params.rp=params.rp/2
new_flux=m.light_curve(params)

#Task2C - setting the radius of the thrid planet to 2*R
params.rp=4*params.rp               #multiplying by 4 because on the previous step we divided the original radius by 2
new_flux2=m.light_curve(params)

fig = plt.figure(figsize=(10, 6))
plt.plot(time, flux, color="teal", label="$R_p$")
plt.plot(time, new_flux, color="limegreen", label="$\\frac{R_{p}}{2}$")
plt.plot(time, new_flux2, color="indianred", label="$2R_p$")
plt.title("Light curves of two exoplanets with radii $R_p$ and $\\frac{R_{p}}{2}$")
plt.legend()
plt.xlabel("Time from transit center (Julian days)")
plt.ylabel("Relative flux")
plt.show()

#fig.savefig('/root/2_veneti_e_mezzo/src/daneel/detection/assignment2_taskB.png', dpi=fig.dpi)  #task 2B
fig.savefig('/root/2_veneti_e_mezzo/src/daneel/detection/assignment2_taskC.png', dpi=fig.dpi)  #task 2C