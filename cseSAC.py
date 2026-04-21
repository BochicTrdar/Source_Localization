import numpy as np
from scipy.io import loadmat
from scipy.signal.windows import hamming
import matplotlib.pyplot as plt
import os

def cseSAC(file, npfft, freqv, vchan, nsnap, noverlap, ifile, ispt):
    """
    Compute estimated cross-spectral density matrices
    from real baseband SACLANT array data stored in mat-files
    
    Parameters:
    -----------
    file : str
        Basic file name of DREA mat-files
    npfft : int
        Length of snapshot in samples
    freqv : array_like
        Normalized frequency vector (to sampling rate of 500 Hz)
    vchan : array_like
        Vector of channel numbers to process
    nsnap : int
        Number of snapshots to average
    noverlap : int
        Number of samples to overlap snapshots
    ifile : int
        mat-file file number
    ispt : int
        starting sample point in file
        
    Returns:
    --------
    Rf : ndarray
        Cross-spectral density matrices
    ifile : int
        Updated file number
    ispt : int
        Updated starting point
    """
    
    j = 1j
    tpi = 2 * np.pi
    msens = len(vchan)
    fmin = freqv[0]
    nfreq = len(freqv)
    ismin = 1  # Is this correct?
    
    # Define file parameters
    fsr = 1000  # 1000 Hz sampling rate
    nchan = 48  # 48 channels of data
    lfils0 = 65536  # length of file in samples/channel
    lfils = lfils0
    nrec = lfils / npfft  # integer number of snapshots/file (>=1)
    ibin1 = int(np.floor(npfft * fmin)) + 1  # starting bin to process
    wh = hamming(npfft, sym=False)  # Hamming window data segments
    
    # Load first required data file
    fnum = f"{ifile}"
    fname = file + fnum
    print(f'Loading the SACLANT mat-file: {fname}')
    
    # Load mat file
    mat_data = loadmat(fname)
    dat = mat_data['dat']
    
    # Correct channel 24 inversion and transpose
    # Note: MATLAB indices are 1-based, Python is 0-based
    # Original: dat = [dat(1:23,:); -dat(24,:); dat(25:48,:)].'
    dat_first = dat[0:23, :]  # rows 1-23
    dat_24 = -dat[23:24, :]    # row 24
    dat_rest = dat[24:48, :]   # rows 25-48
    dat = np.vstack((dat_first, dat_24, dat_rest)).T
    
    ldat = dat.shape[0]  # length of data record
    
    # Set starting point if required to get signal or noise only data
    # Note: No checking done to make sure nsnap not greater than noise
    # only or signal data!
    if ispt == 0:  # signal + noise case
        ispt = ismin
    elif ispt == -1:  # noise only desired
        # ismax is not defined in the original code, assuming it exists
        # You may need to define ismax appropriately
        ismax = ldat // 2  # Example assumption
        if ismin > ldat - ismax:
            ispt = 1  # start before signal
        else:
            ispt = ismax + 1  # start after signal
    
    # Start forming cross-spectral density matrices
    Rf = np.zeros((msens * nfreq, msens), dtype=complex)
    x = np.zeros((npfft, msens), dtype=complex)
    
    for isnap in range(nsnap):
        iend = ispt + npfft - 1  # index of last point in current snapshot
        
        # Load data file if required
        if iend > lfils:
            dsav = dat[ispt-1:lfils, :]  # -1 for Python indexing
            ifile += 1
            fnum = f"{ifile}"
            fname = file + fnum
            print(f'Loading the SACLANT mat-file: {fname}')
            mat_data = loadmat(fname)
            dat = mat_data['dat']
            
            # Correct channel 24 inversion and transpose
            dat_first = dat[0:23, :]
            dat_24 = -dat[23:24, :]
            dat_rest = dat[24:48, :]
            dat = np.vstack((dat_first, dat_24, dat_rest)).T
            
            dat = np.vstack((dsav, dat))  # prepend data from last file
            lfils = lfils0 + lfils - ispt + 1
            ispt = 1
        
        # Take Hamming windowed fft's of each sensor output
        for isens in range(msens):
            ichan = vchan[isens] - 1  # -1 for Python indexing
            ipt1 = ispt - 1  # -1 for Python indexing
            ipt2 = ispt + npfft - 1  # ipt2 is already 1-based, no -1 needed
            windowed_data = wh * dat[ipt1:ipt2, ichan]
            x[:, isens] = np.fft.fft(windowed_data, npfft)
        
        # Form csdm estimate for frequency band of interest
        ibin = ibin1
        for ib in range(nfreq):
            freq = fsr * (ibin - 1) / npfft
            y = x[ibin-1, :].reshape(-1, 1)  # -1 for Python indexing
            y = y / np.linalg.norm(y)  # Normalize
            
            # Accumulate csdm estimate
            k1 = ib * msens
            k2 = (ib + 1) * msens
            Rf[k1:k2, :] = Rf[k1:k2, :] + (y @ y.conj().T) / nsnap
            ibin += 1
        
        ispt = ispt + npfft - noverlap
    
    return Rf, ifile, ispt


# Example usage (commented out):
"""
# Define parameters
file = 'saclant_data_'
npfft = 1024
freqv = np.array([0.1, 0.2, 0.3])  # Example normalized frequencies
vchan = np.arange(1, 49)  # Channels 1-48
nsnap = 10
noverlap = 512
ifile = 1
ispt = 0

# Call the function
Rf, new_ifile, new_ispt = cseSAC(file, npfft, freqv, vchan, nsnap, noverlap, ifile, ispt)
"""
