# Lab Report: ECG and HRV Analysis

**Project**: ECG Signal Processing and Heart Rate Variability (HRV) Dashboard
**Date**: April 2026

## 1. Methodology

The goal of this project was to establish an automated, robust signal processing pipeline to analyze single-lead Electrocardiogram (ECG) data and extract Heart Rate Variability (HRV) metrics. The implementation involves digital filtering, feature extraction, and statistical/mathematical analysis.

### 1.1 Preprocessing and Filtering
Raw ECG signals often contain noise artifacts, including baseline wander (due to respiration or patient movement) and high-frequency noise (from muscle contractions or power-line interference). 
To mitigate these, a **Butterworth Bandpass Filter** ($0.5$ Hz to $40.0$ Hz, 3rd order) was applied. 
- The lower cutoff ($0.5$ Hz) effectively removes baseline wander without distorting the clinical characteristics of the ST segment.
- The upper cutoff ($40.0$ Hz) attenuates high-frequency noise and the $50/60$ Hz power-line interference.
The filter was implemented using a zero-phase forward and reverse digital IIR filter (`scipy.signal.filtfilt`) to prevent phase shift or delay in the signal, ensuring that peak detection remains temporally accurate.

### 1.2 R-Peak Detection
Accurate localization of the R-peaks (the most prominent feature of the QRS complex) is crucial for HRV analysis. The algorithm we implemented mirrors the foundational Pan-Tompkins method:
1. **Differentiation**: The filtered signal was differentiated to emphasize the high-slope parts of the QRS complex.
2. **Squaring**: The derivative was squared point-by-point to make all data points positive and to non-linearly amplify higher frequencies (i.e., the QRS complexes) while suppressing lower-frequency P and T waves.
3. **Moving Window Integration**: We passed the squared signal through a moving average window (approx $150$ ms). This extracts the energy waveform of the QRS.
4. **Thresholding & Peak Finding**: A dynamic threshold (based on a percentage of the maximum amplitude) was used alongside a minimum distance criterion ($0.4$ seconds) to identify the R-peaks robustly in the integrated signal. The exact location is refined by searching for the local maxima in the original filtered signal within a small radius.

### 1.3 HRV Parameter Calculation

Using the detected R-peaks, sequences of RR intervals were computed (the time between successive R-peaks, usually in milliseconds).

#### Time-Domain Analysis
Time-domain features are basic statistical measures of the continuously measured intervals:
- **SDNN**: Standard Deviation of NN (normal-to-normal RR) intervals. It estimates overall HRV.
- **RMSSD**: Root Mean Square of Successive Differences. It is the primary time-domain measure used to estimate the vagally mediated changes reflected in HRV.

#### Non-Linear Analysis
Because the cardiovascular system is highly complex and non-linear:
- **Poincaré Map**: A scatter plot of the current RR interval ($RR_n$) against the subsequent one ($RR_{n+1}$). Features $SD1$ (short-term variability) and $SD2$ (long-term variability) are derived from the minor and major axes of the ellipse fitted to this map.
- **Sample Entropy (SampEn)**: Quantifies the regularity and complexity of the signal. Lower values indicate more self-similarity (less complex), while higher values denote more randomness.

#### Frequency-Domain Analysis
Frequency analysis was conducted using Welch's method for Power Spectral Density (PSD) estimation:
1. The discrete RR intervals were interpolated using linear interpolation and resampled at a constant frequency ($4$ Hz) to allow the use of Fast Fourier Transform methods.
2. We then computed absolute spectral power in two primary bands:
   - **LF (Low Frequency)**: $0.04 - 0.15$ Hz.
   - **HF (High Frequency)**: $0.15 - 0.40$ Hz.
3. The **LF/HF Ratio** was subsequently calculated.

---

## 2. Results

The processing pipeline is wrapped within an interactive dashboard utilizing Streamlit and Plotly. The visual output includes:
1. A synchronized plot of the **Raw ECG vs Filtered ECG**, where automatically identified R-peaks are superimposed as red markers. The alignment demonstrates that the algorithm accurately avoids T-waves and localizes the maximum point of ventricular depolarization.
2. An **RR Interval Tachogram**, plotting interval duration against time.
3. A **Poincaré Plot** depicting the correlation between adjacent RR intervals, heavily centered along the line of identity.
4. The **Power Spectral Density** graph, visually distinguishing the LF ($0.04 - 0.15$ Hz, shaded yellow) and HF ($0.15 - 0.4$ Hz, shaded green) bands.

All calculated metric values (SDNN, RMSSD, Power bands, SD1, SD2, Sample Entropy) are correctly output to the metric displays.

---

## 3. Discussion

The heart's rhythm is largely governed by the autonomic nervous system (ANS) which consists of two antagonist branches: the sympathetic nervous system ("fight or flight", which accelerates heart rate) and the parasympathetic or vagal nervous system ("rest and digest", which decelerates heart rate). Heart Rate Variability is a non-invasive window into this autonomic regulation.

### Interpretation of Time and Non-Linear Domains
Variables like RMSSD and SD1 (from the Poincaré plot) predominantly reflect the high-frequency variations in heart rate. Since parasympathetic nerve traffic can modulate heart rate significantly faster than sympathetic traffic, RMSSD and SD1 are primarily indexes of *parasympathetic* tone. A low RMSSD is generally associated with increased risk in cardiovascular conditions or high stress. Sample Entropy indicates the overall unpredictability of the time series; a healthier physiological system generally displays high complexity (higher entropy), whereas illness or extreme stress often reduces this complexity.

### The Autonomic Balance in the Frequency Domain
Frequency-domain parameters offer a deeper look at the interplay of the ANS:
- **HF Power**: This band corresponds to respiratory sinus arrhythmia (RSA) and is driven almost exclusively by parasympathetic (vagal) activity.
- **LF Power**: The interpretation of LF power is more nuanced; however, it is generally accepted that it reflects a combination of both sympathetic and parasympathetic activity (e.g., related to blood pressure regulation and baroreceptor activity).
- **LF/HF Ratio**: This ratio is classically utilized as a marker of sympatho-vagal balance. A high LF/HF ratio typically implies sympathetic dominance (such as during exercise, mental stress, or pathological states like heart failure), whereas a low ratio implies parasympathetic dominance (observed during rest and relaxation). Although recent physiological debates suggest the interaction is more complex than a simple see-saw balance, tracing the LF/HF ratio remains a valuable heuristic for evaluating generalized physiological arousal.

Through this dashboard, researchers or clinicians can quickly quantify this balance and visually examine the raw waveforms contributing to the calculations, confirming the reliability of automatic inferences.
