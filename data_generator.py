import pandas as pd
import numpy as np
import scipy.datasets

def generate_sample_data(filename='sample_ecg.csv', duration_seconds=60):
    """
    Generates a sample ECG dataset using SciPy's built-in electrocardiogram dataset.
    The original data is sampled at 360 Hz.
    """
    fs = 360  # Hz
    print(f"Loading SciPy electrocardiogram dataset...")
    ecg_signal = scipy.datasets.electrocardiogram()
    
    # Take a 60-second snippet to keep file size small but enough for HRV
    # A full 5 minutes is available, but 60 seconds is good for a quick demo
    num_samples = fs * duration_seconds
    if num_samples > len(ecg_signal):
        num_samples = len(ecg_signal)
        
    ecg_snippet = ecg_signal[:num_samples]
    time = np.arange(num_samples) / fs
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time (s)': time,
        'ECG (mV)': ecg_snippet
    })
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Saved {duration_seconds} seconds of ECG data to {filename}")

if __name__ == '__main__':
    generate_sample_data()
