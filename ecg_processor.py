import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.interpolate import interp1d
import antropy as ant

def filter_ecg(signal, fs, lowcut=0.5, highcut=40.0, order=3):
    """
    Apply a Butterworth bandpass filter to the ECG signal to remove 
    baseline wander (low frequency) and high-frequency noise.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # We must restrict 'high' to be strictly less than 1.0
    if high >= 1.0:
        high = 0.99
        
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def find_r_peaks(signal, fs, threshold_ratio=0.6, distance_sec=0.4):
    """
    A simplified Pan-Tompkins style algorithm for R-peak detection.
    Returns a dictionary containing the r_peaks and the intermediate 
    signals (derivative, squared, integrated) for visualization.
    """
    # 1. Bandpass 5-15Hz (Standard Pan-Tompkins bandpass range)
    # We will compute it here just for the peak detection to be true to Pan-Tompkins
    nyq = 0.5 * fs
    b, a = butter(1, [5.0/nyq, 15.0/nyq], btype='band')
    pt_filtered = filtfilt(b, a, signal)

    # 2. Differentiation
    diff_sig = np.diff(pt_filtered)
    diff_sig = np.append(diff_sig, 0) # matching length
    
    # 3. Squaring
    squared_sig = diff_sig ** 2
    
    # 4. Moving Window Integration (approx 150 ms)
    window_length = int(0.15 * fs)
    if window_length == 0:
        window_length = 1
        
    mwi_sig = np.convolve(squared_sig, np.ones(window_length)/window_length, mode='same')
    
    # 5. Find Peaks
    threshold = np.max(mwi_sig) * threshold_ratio
    distance = int(distance_sec * fs)  # Minimum distance between peaks
    
    peaks_mwi, _ = find_peaks(mwi_sig, height=threshold, distance=distance)
    
    # Refine peak locations on the original signal (search in a small radius around MWI peak)
    radius = int(0.1 * fs) # 100ms
    r_peaks = []
    for peak in peaks_mwi:
        start = max(0, peak - radius)
        end = min(len(signal), peak + radius)
        local_max_idx = start + np.argmax(signal[start:end])
        # Add only if it's not a duplicate
        if local_max_idx not in r_peaks:
            r_peaks.append(local_max_idx)
            
    r_peaks = np.sort(r_peaks)
    
    return {
        'r_peaks': r_peaks,
        'filtered': pt_filtered,
        'derivative': diff_sig,
        'squared': squared_sig,
        'mwi': mwi_sig,
        'mwi_peaks': peaks_mwi,
        'threshold': threshold
    }

def compute_rr_intervals(peaks, fs):
    """
    Compute RR intervals in milliseconds.
    """
    rr_intervals_sec = np.diff(peaks) / fs
    rr_intervals_ms = rr_intervals_sec * 1000
    return rr_intervals_ms

def calculate_time_domain(rr_intervals_ms):
    """
    Calculate Time-Domain HRV parameters.
    """
    if len(rr_intervals_ms) < 2:
        return {'SDNN': np.nan, 'RMSSD': np.nan, 'Mean RR': np.nan, 'Mean HR': np.nan}
        
    sdnn = np.std(rr_intervals_ms, ddof=1)
    
    diff_rr = np.diff(rr_intervals_ms)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    
    mean_rr = np.mean(rr_intervals_ms)
    mean_hr = 60000 / mean_rr if mean_rr > 0 else 0
    
    return {
        'SDNN (ms)': sdnn,
        'RMSSD (ms)': rmssd,
        'Mean RR (ms)': mean_rr,
        'Mean HR (bpm)': mean_hr
    }

def calculate_frequency_domain(rr_intervals_ms, peaks, fs):
    """
    Calculate Frequency-Domain HRV parameters using Welch's method.
    """
    if len(rr_intervals_ms) < 5:
        return {'LF': np.nan, 'HF': np.nan, 'LF/HF Ratio': np.nan, 'freqs': [], 'psd': []}
        
    # Time corresponding to each RR interval (we use the location of the 2nd peak)
    # Convert peaks to seconds
    times_sec = peaks[1:] / fs
    
    # Resample RR intervals (typically 4 Hz)
    fs_resample = 4.0
    times_resample = np.arange(times_sec[0], times_sec[-1], 1/fs_resample)
    
    # Linear interpolation
    interp_func = interp1d(times_sec, rr_intervals_ms, kind='linear', fill_value='extrapolate')
    rr_resampled = interp_func(times_resample)
    
    # Remove mean
    rr_resampled = rr_resampled - np.mean(rr_resampled)
    
    # Welch's Method
    # Appropriate segment length (e.g., 256 samples, which is ~64 secs at 4 Hz)
    nperseg = min(256, len(rr_resampled))
    if nperseg < 4:
        return {'LF': np.nan, 'HF': np.nan, 'LF/HF Ratio': np.nan, 'freqs': [], 'psd': []}
        
    freqs, psd = welch(rr_resampled, fs=fs_resample, nperseg=nperseg, scaling='density')
    
    # Define frequency bands
    vlf_band = (0.0033, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.40)
    
    # Find indices
    lf_idx = np.where((freqs >= lf_band[0]) & (freqs < lf_band[1]))[0]
    hf_idx = np.where((freqs >= hf_band[0]) & (freqs < hf_band[1]))[0]
    
    # Calculate absolute power (integrate using trapezoidal rule or simply sum if df is constant)
    df = freqs[1] - freqs[0]
    lf_power = np.sum(psd[lf_idx]) * df
    hf_power = np.sum(psd[hf_idx]) * df
    
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan
    
    return {
        'LF Power': lf_power,
        'HF Power': hf_power,
        'LF/HF Ratio': lf_hf_ratio,
        'freqs': freqs.tolist(),
        'psd': psd.tolist()
    }

def calculate_nonlinear(rr_intervals_ms):
    """
    Calculate Non-Linear HRV parameters (Poincaré and Entropy).
    """
    if len(rr_intervals_ms) < 3:
        return {'SD1': np.nan, 'SD2': np.nan, 'Sample Entropy': np.nan}
        
    # Poincaré map features
    rr_n = rr_intervals_ms[:-1]
    rr_n1 = rr_intervals_ms[1:]
    
    sd1 = np.sqrt(0.5) * np.std(rr_n1 - rr_n, ddof=1)
    sd2 = np.sqrt(0.5) * np.std(rr_n1 + rr_n, ddof=1)
    
    # Sample Entropy
    try:
        samp_en = ant.sample_entropy(rr_intervals_ms)
    except Exception:
        samp_en = np.nan
        
    mean_rr_n = np.mean(rr_n) if len(rr_n) > 0 else np.nan
    mean_rr_n1 = np.mean(rr_n1) if len(rr_n1) > 0 else np.nan
    
    return {
        'SD1': sd1,
        'SD2': sd2,
        'Sample Entropy': samp_en,
        'rr_n': rr_n.tolist(),
        'rr_n1': rr_n1.tolist(),
        'mean_rr_n': mean_rr_n,
        'mean_rr_n1': mean_rr_n1
    }
