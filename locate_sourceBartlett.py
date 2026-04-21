# ==================================================================
#  
#  KRAKEN 169 Hz: static source localization
#  Faro, ter 31 mar 2026 10:51:50 
#  Written by Tordar 
#  
# ==================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import warnings
import sys
# Import cseSAC function
from cseSAC import cseSAC

warnings.filterwarnings('ignore')

# Load KRAKEN replicas
print('Loading replicas...')
data = loadmat('replica_cpr.mat')  # Adjust variable names as needed

# Assuming the .mat file contains variables: p, zs, rarray
# You may need to check the actual variable names in your .mat file
p = data['p']          # Replica pressure field

zs = data['zs'].flatten()  # Source depths
rarray = data['rarray'].flatten()  # Ranges

rarraykm = rarray / 1000

nzs = len(zs)
nra = len(rarray)

Rmin = min(rarray)
Rmax = max(rarray)
zsmin = min(zs)
zsmax = max(zs)

print('Generating the covariance matrix...')

freqv = np.array([169.0/1000])  # kHz
npfft = 2**9  # 512
vchan = np.arange(1, 49)  # 1:48
nhyds = len(vchan)
noverlap = 0
ispt = 1

# Keep the max, to compare cases
ifile = 1  # (5500,76) m 0.81950 <= 
#ifile = 2  # (5500,76) m 0.81598
#ifile = 3  # (5500,76) m 0.80055
#ifile = 4  # (5500,76) m 0.80320
#ifile = 5  # (5500,75) m 0.80348
#ifile = 6  # (5500,75) m 0.80456
#ifile = 7  # (5500,75) m 0.81275
#ifile = 8  # (5500,75) m 0.80487
#ifile = 9  # (5500,75) m 0.79703
#ifile = 10 # (5500,75) m 0.80558

nsnap = int(65536 / npfft)

if ifile in [5, 10]:
    nsnap = int(32768 / npfft)

# Call cseSAC function
R, ifile, ispt = cseSAC('A2601_1/A2601_1_', 
                        npfft, freqv, vchan, nsnap, noverlap, ifile, ispt)

# Initialize Bartlett processor
bartlett = np.zeros((nzs, nra))

print('Locating the source...')

for i in range(nzs):
    
    # Extract replica field for this source depth
    # p has dimensions [nzs, nra, nhyds] or [nzs, nra, nhyds]
    # p_i = squeeze(p(i,:,:)) -> shape (nra, nhyds)
    p_i = p[i, :, :]  # Adjust indexing if dimensions are different
    for j in range(nra):
        
        # Extract replica for this range
        e = p_i[:, j].copy()
        e = np.flipud(e)  # THIS IS VERY IMPORTANT!!!!
        e = e / np.linalg.norm(e)
        eR = np.dot(np.conj(e),R)
        bartlett[i,j] = np.abs( np.dot(eR,e) )

# Find maximum
I, K = np.unravel_index(np.argmax(bartlett), bartlett.shape)

print('\nDisplaying results:')

thetitle = f'Bartlett @ 169 Hz: source located at ({rarray[K]},{zs[I]}) m'

# Plot results
plt.figure(1,dpi=300)
plt.pcolormesh(rarraykm, zs, bartlett, shading='interp')
plt.colorbar(label='Bartlett Power')
plt.plot(rarraykm[K], zs[I], 'wo', markersize=15, linewidth=2)
plt.axis([rarraykm[0], rarraykm[-1], zs[0], zs[-1]])
plt.gca().invert_yaxis()  # Equivalent to view(0,-90)
plt.xlabel('Range (km)')
plt.ylabel('Depth (m)')
plt.title(thetitle)
plt.tight_layout()
plt.show()

maxb = np.max(bartlett)
theposition = f'({rarray[K]},{zs[I]}) m {maxb:.5f}'
print(theposition)

print('done.')
