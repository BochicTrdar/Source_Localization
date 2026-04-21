# GA mean parameters according to Gingras1995
# Attenuation in dB/lambda

Rmax = 11000
zs = 74.6
Dmax = 128.9

thickness = 3.3
cp = [1505, 1556, 1576]
ap = [0.11, 0.18]
rho = [2.0, 1.6]

first_hyd = 17.7
dhyd = 2
last_hyd = 111.7

rarray = 5437
zarray = [i for i in range(int(first_hyd), int(last_hyd) + 1, int(dhyd))]
# More precise way to create zarray:
# zarray = list(range(int(first_hyd), int(last_hyd) + int(dhyd), int(dhyd)))
# Or using numpy for floating point precision:
# import numpy as np
# zarray = np.arange(first_hyd, last_hyd + dhyd, dhyd)

nza = len(zarray)

freq = 169
w = 2*np.pi*freq
