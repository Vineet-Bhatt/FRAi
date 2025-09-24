import numpy as np
from scipy.signal import savgol_filter

def resample_to_grid(freqs, mags, target_grid):
    return np.interp(target_grid, freqs, mags)

def smooth(signal, window=11, poly=3):
    if len(signal) < window:
        return signal
    return savgol_filter(signal, window_length=window, polyorder=poly)

def normalize(signal):
    # zero mean and unit std
    mean = np.mean(signal)
    std = np.std(signal) + 1e-9
    return (signal - mean)/std

def featurize(freqs, mags):
    # Example: return vector of magnitudes on common grid plus some derived features
    deriv = np.gradient(mags, freqs)
    peaks = mags[(np.r_[True, mags[1:] > mags[:-1]] & np.r_[mags[:-1] > mags[1:], True])]
    return {
        'mags': mags,
        'deriv': deriv,
        'peaks_count': len(peaks),
        'mean': np.mean(mags),
        'std': np.std(mags)
    }
