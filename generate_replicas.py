# ==================================================================
#  
#  KRAKEN: source localization, replicas
#  Faro, ter 31 mar 2026 10:48:43 
#  Written by Tordar 
#  
# ==================================================================
# Sorry, no critical angle and number of modes because cp < cw... 

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.interpolate import interp1d
import os
import sys
import subprocess
import warnings
from read_modes    import * 
from read_shd      import *
from wkrakenenvfil import *

warnings.filterwarnings('ignore')

# Load geometry parameters
exec(open('gamean_geometry.py').read())
case_title = 'SACLANT'
Rmin = 4000
Rminkm = Rmin / 1000
Rmax = 7000
Rmaxkm = Rmax / 1000
nra = 101
rarray = np.linspace(Rmin, Rmax, nra)
rarraykm = rarray / 1000
# ==================================================================
#  
#  Source properties
#  
# ==================================================================

zs = np.arange(1, Dmax)  # 1:Dmax-1
nzs = len(zs)

source_data = {
    'n': nzs,
    'zs': zs,
    'f': freq
}

# ==================================================================
#  
#  Surface properties
#  
# ==================================================================

surface_data = {
    'bc': 'V',
    'properties': [],  # Not required due to vacuum over surface
    'reflection': []   # Not required for this case
}

# ==================================================================
#  Scatter is not required for this case
# ==================================================================

scatter_data = {
    'bumden': [],  # Bump density in ridges/km
    'eta'   : [],     # Principal radius 1 of bump
    'xi'    : []       # Principal radius 2 of bump
}

# ==================================================================
#  
#  Sound speed profile properties
#  
# ==================================================================

nmesh = 0

# Load ssp.dat
ssp = np.loadtxt('saclant.ssp')
zi = ssp[:, 0]
ci = ssp[:, 1]

# Add bottom
zi = np.append(zi, Dmax)
ci = np.append(ci, ci[-1])

# Add surface if needed
if zi[0] > 0:
    zi = np.insert(zi, 0, 0)
    ci = np.insert(ci, 0, ci[0])
cmin = np.min( ci )
kw = w/cmin 

Etter_threshold = np.floor(10*cmin/Dmax)

csw  = np.zeros_like(zi)
rhow = np.ones_like( zi)
apw  = np.zeros_like(zi)
asw  = np.zeros_like(zi)

ssp_data = {
    'type': 'H',
    'itype': 'N',
    'nmesh': nmesh,  # Number of mesh points (about 10 per vertical wavelength)
    'sigma': 0,      # RMS roughness at the surface
    'clow': 0.0,
    'chigh': 5000.0,
    'cdata': np.vstack([zi, ci, csw, rhow, apw, asw]),
    'zbottom': Dmax
}

# ==================================================================
#  
#  Bottom properties 
#  
# ==================================================================

layer_info = np.array([[Dmax          ,cp[0],0,rho[0],ap[0],0], # [z,cp,cs,RHO,ap,as]
                    [Dmax+thickness,cp[2],0,rho[1],ap[1],0]])

layerp     = np.array([[nmesh, 0.0, Dmax+thickness]])
layert     = 'HH'
properties = np.array( layer_info[1,] )

m1 = np.array( [layer_info[0,],layer_info[0,]] )

bdata        = np.array( [m1] )
bdata[0,1,0] = layer_info[1,0]

units      = 'W';
bc	   = 'A';
sigma      = 0.0 # Interfacial roughness

bottom_data = {"n":2,"layerp":layerp,"layert":layert,"properties":properties,"bdata":bdata,"units":units,"bc":bc,"sigma":sigma}
# ==================================================================
#  
#  Field definition
#  
# ==================================================================

nza = len(zarray)

field_data = {
    'rmax': Rmaxkm,
    'rr': rarraykm,
    'nrr': nra,
    'rp': 0,  # Range of first profile
    'np': 1,  # Number of profiles
    'm': 999,
    'rmodes': 'A',
    'stype': 'R',
    'thorpe': 'T',
    'finder': ' ',
    'rd': zarray,
    'dr': np.zeros_like(zarray),
    'nrd': nza
}

print('Generating replicas...')

# ==================================================================
#  
#  Write the input file for KRAKEN
#  
# ==================================================================

# Write environment file
wkrakenenvfil('saclant', case_title, source_data, surface_data, 
              scatter_data, ssp_data, bottom_data, field_data)

# ==================================================================
#  
#  Run the model:
#  
# ==================================================================

print('Running KRAKEN...')
subprocess.run(['krakenc.exe', 'saclant'])

Modes = read_modes( 'saclant.mod', freq )
k   = Modes[ 'k' ]
Phi = Modes['phi']
z   = Modes[ 'z' ]
dz  = np.diff( z )
cos_thetam = np.real( k )/kw # In the general case check out for values larger than 1... 
thetam = np.arccos( cos_thetam )
thetamd = np.degrees( thetam )
#print( thetam )
nmodes = len( thetam )
print( 'M =', nmodes,'modes' ) 
kh = kw*np.cos( thetam ) #  Horizontal wavenumber
kv = kw*np.sin( thetam ) #    Vertical wavenumber
mode_cycle_distance = -2*np.pi/np.diff( kh )
ray_cycle_distance  =  2*Dmax/np.tan( thetam[0:-1] )
# ==================================================================
# Reflection coefficient for modes 
# (not accurate after critical angle): 
R_normal_modes = np.zeros(nmodes) + 1j*np.zeros(nmodes)
for m in range(nmodes):
    phim = Phi[:,m]
    phiD = phim[-1];
    dphidz = np.diff( phim )/dz
    dphidzD = dphidz[-1]
    r = phiD/dphidzD
    E = np.exp( 2*1j*kv[m]*Dmax );
    R_normal_modes[m] = ( 1 + 1j*kv[m]*r )/( E*( 1j*kv[m]*r - 1 ) )
# ==================================================================
# Turning/Bouncing points:
zdown = np.zeros(nmodes)      # Close to the surface
zup   =  np.ones(nmodes)*Dmax # Close to the bottom - no need to care about it... 
ISnell = cos_thetam/cmin
cray = 1/ISnell
for imode in range(nmodes):
    dc = ci - cray[imode]
    amax = np.max( dc )
    amin = np.min( dc )
    if ( amax*amin < 0 ): # A sign change indicates that there is a turning depth
        pos = np.argmin( np.abs(dc) )
        zdown[imode] = np.interp(0, dc[pos-2:pos+3],zi[pos-2:pos+3]) # No need to interpolate over the entire interval...
# ==================================================================
#  
#  Waveguide invariant stuff:
#  
# ==================================================================
k1 = k
rk1 = np.real( k1 )
nk1 = len( k1 )
df = 1
freq2 = freq + df
source_data = {
    'n': nzs,
    'zs': zs,
    'f': freq2
}

wkrakenenvfil('invariant',case_title,source_data,surface_data,scatter_data,ssp_data, bottom_data,field_data);
subprocess.run(['krakenc.exe', 'invariant'])
Modes = read_modes( 'invariant.mod', freq )
subprocess.run(['rm', 'invariant.mod']) 
k2   = Modes['k']
rk2 = np.real( k2 ) 
nk2 = len( k2 )
nk = min([nk1,nk2]);
k  = 0.5*( rk1[0:nk]+rk2[0:nk] )
dk = rk2[0:nk] - rk1[0:nk] 
phase_speed = w/k 
phase_slowness = 1/phase_speed
group_speed = 2*np.pi*df/dk 
group_slowness = 1/group_speed
coeffs_polyfit = np.polyfit(group_slowness, phase_slowness, 1)
thebeta = -coeffs_polyfit[0]
#print( thebeta )
# ==================================================================
#  
#  Get the field:
#  
# ==================================================================

subprocess.run(['mv', 'field.flp', 'saclant.flp'])
subprocess.run(['field.exe', 'saclant','saclant.flp'])

# ==================================================================
#  
#  Read the field:
#  
# ==================================================================

PlotTitle, PlotType, freqVec, freq0, atten, Pos, p = read_shd('saclant')
p = np.squeeze(p)  # Remove singleton dimensions

# ==================================================================
#  
#  Save replicas
#  
# ==================================================================

# Save as MATLAB .mat file
savemat('replica_cpr.mat', {
    'p': p,
    'rarray': rarray,
    'zs': zs
})

print(Etter_threshold,freq)
print('beta = ',thebeta)

plt.figure(1,dpi=300)
plt.plot(thetamd,abs(R_normal_modes),'ko')
plt.xlabel('Grazing Angle (degrees)')
plt.ylabel(r'$|R(\theta)|$')
plt.grid('on')

plt.figure(2,dpi=300)
plt.plot(group_slowness,phase_slowness)
plt.xlabel('Group Slowness (s/m)')
plt.ylabel('Phase Slowness (s/m)')
plt.grid('on')

plt.show()

subprocess.run(['rm', 'invariant.env', 'invariant.prt']) 

print('done.')
