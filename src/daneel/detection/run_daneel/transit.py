import batman
import numpy as np
import matplotlib.pyplot as plt
import parameters as p

time = np.linspace(2454966.6983 - 1, 2454966.6983 + 1, 1000) # time array in days (centered on transit)

params = batman.TransitParams()
params.t0 = p.t0
params.per = p.per
params.rp = p.rp
params.a = p.a
params.inc = p.inc
params.ecc = p.ecc
params.w = p.w
params.limb_dark = p.limb_dark
params.u = p.u

m = batman.TransitModel(params, time)    #initializes model
flux = m.light_curve(params)           # calculates light curve

fig=plt.figure(figsize=(10, 6))
plt.plot(time, flux, color="salmon", label="Model Light Curve")
plt.xlabel("Time from transit center (Julian days)")
plt.ylabel("Relative flux")
plt.title("Kepler-22 b Transit Light Curve")
plt.legend()
plt.show()


plt.savefig("transit_curve_Eslamifar.png", dpi=300, bbox_inches="tight")