import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from ecg_processor import (
    filter_ecg,
    find_r_peaks,
    compute_rr_intervals,
    calculate_time_domain,
    calculate_frequency_domain,
    calculate_nonlinear
)

st.set_page_config(page_title="ECG & HRV Analysis Dashboard", layout="wide")

st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h2>Riphah International University, Lahore</h2>
    <h4>Faculty of Engineering & Applied Sciences</h4>
    <h4>Department of Biomedical Engineering</h4>
    <p style="font-size: 18px;">
        <b>Program:</b> B.Sc. Biomedical Engineering &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; <b>Semester:</b> VI <br>
        <b>Subject:</b> Biomedical Signal Processing &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; <b>Experiment #:</b> 5
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

col_name, col_sap = st.columns(2)
with col_name:
    st.text_input("Name:", placeholder="Enter your Name")
with col_sap:
    st.text_input("SAP ID:", placeholder="Enter your SAP ID")

st.markdown("<hr>", unsafe_allow_html=True)
st.title("Open Ended Lab 1: ECG & HRV Analysis Dashboard")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration")

# 1. Data Upload
st.sidebar.subheader("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload ECG Data (.csv, .mat, .dat)", type=['csv', 'mat', 'dat'])

# We default to the generated sample file if none is uploaded
data_source = uploaded_file if uploaded_file is not None else "sample_ecg.csv"

@st.cache_data
def load_data(source):
    import scipy.io as sio
    
    # helper to process dataframe
    def process_df(df):
        return df

    if isinstance(source, str):
        if not os.path.exists(source):
            return None
        file_name = source
        file_obj = source
    else:
        file_name = source.name
        file_obj = source

    try:
        if file_name.endswith('.csv'):
            return pd.read_csv(file_obj)
        elif file_name.endswith('.mat'):
            mat = sio.loadmat(file_obj)
            # Find arrays in mat
            arrays = {k: v for k, v in mat.items() if not k.startswith('__') and isinstance(v, np.ndarray)}
            if not arrays:
                st.error("No valid arrays found in the .mat file.")
                return None
            
            # Simple heuristic: try to create a DataFrame out of 1D arrays or the largest 2D array
            df_dict = {}
            for k, v in arrays.items():
                v = np.squeeze(v)
                if v.ndim == 1:
                    df_dict[k] = v
                elif v.ndim == 2:
                    # if it's 2D, we can take the first few columns
                    for i in range(min(v.shape[1], 5)):
                        df_dict[f"{k}_{i}"] = v[:, i]
            
            if df_dict:
                return pd.DataFrame(df_dict)
            else:
                st.error("Could not parse arrays from .mat file into columns.")
                return None
                
        elif file_name.endswith('.dat'):
            # Try comma or whitespace separated
            try:
                if isinstance(source, str):
                    return pd.read_csv(file_obj, sep=r'\s+')
                else:
                    return pd.read_csv(file_obj, sep=r'\s+')
            except Exception:
                # If that fails, try generic loadtxt
                data = np.loadtxt(file_obj)
                if data.ndim == 1:
                    return pd.DataFrame({'Data': data})
                else:
                    return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None
    return None

df = load_data(data_source)

if df is not None:
    # Column selection
    columns = df.columns.tolist()
    ecg_col = st.sidebar.selectbox("Select ECG Column", options=columns, index=min(1, len(columns)-1))
    
    # Try to find a time column, otherwise generate one
    time_col = None
    for col in columns:
        if 'time' in str(col).lower() or 't' == str(col).lower():
            time_col = col
            break
            
    if time_col:
        # Convert to numeric, forcing non-numeric (like string dates) to NaN
        time_data = pd.to_numeric(df[time_col], errors='coerce').values
        
        # If we successfully parsed some numeric time
        if not np.all(np.isnan(time_data)):
            # find valid dt from the first two valid samples
            valid_idx = np.where(~np.isnan(time_data))[0]
            if len(valid_idx) > 1:
                dt = time_data[valid_idx[1]] - time_data[valid_idx[0]]
                if dt > 0:
                    default_fs = int(np.round(1.0 / dt))
                else:
                    default_fs = 360
            else:
                default_fs = 360
        else:
            time_data = None
            default_fs = 360
    else:
        time_data = None
        default_fs = 360
        
    fs = st.sidebar.number_input("Sampling Rate (Hz)", min_value=1, max_value=2000, value=default_fs)

    # Filtering parameters
    st.sidebar.subheader("Filter Parameters")
    lowcut = st.sidebar.slider("Lowcut Frequency (Hz)", 0.1, 5.0, 0.5, 0.1)
    highcut = st.sidebar.slider("Highcut Frequency (Hz)", 10.0, 100.0, 40.0, 1.0)
    
    # Peak detection parameters
    st.sidebar.subheader("Peak Detection")
    threshold = st.sidebar.slider("Threshold Ratio", 0.1, 1.0, 0.6, 0.05)
    distance = st.sidebar.slider("Min Distance (sec)", 0.1, 1.0, 0.4, 0.1)
    
    # --- PROCESSING ---
    ecg_signal = df[ecg_col].values
    
    if time_data is None:
        time_data = np.arange(len(ecg_signal)) / fs
        
    # 1. Filter
    filtered_ecg = filter_ecg(ecg_signal, fs, lowcut=lowcut, highcut=highcut)
    
    # 2. Peaks
    pt_results = find_r_peaks(filtered_ecg, fs, threshold_ratio=threshold, distance_sec=distance)
    r_peaks = pt_results['r_peaks']
    
    # 3. RR Intervals
    rr_intervals = compute_rr_intervals(r_peaks, fs)
    
    # 4. HRV Metrics
    td_metrics = calculate_time_domain(rr_intervals)
    fd_metrics = calculate_frequency_domain(rr_intervals, r_peaks, fs)
    nl_metrics = calculate_nonlinear(rr_intervals)
    
    # --- MAIN VIEW ---
    st.header("1. Signal Processing & Peak Detection")
    
    tab_raw, tab_pt = st.tabs(["Original vs Filtered ECG", "Pan-Tompkins Algorithm Stages"])
    
    # We plot a subset to prevent WebGL crashing if data is huge, e.g., first 30 seconds
    plot_points = min(len(ecg_signal), 15 * fs) # Using 15 second subset for crisper visuals
    
    with tab_raw:
        # Plot: Raw vs Filtered with Peaks
        fig_signal = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                   subplot_titles=("Raw ECG", "Filtered ECG with R-Peaks"))
                                   
        # Raw
        fig_signal.add_trace(go.Scatter(x=time_data[:plot_points], y=ecg_signal[:plot_points], 
                                        mode='lines', name='Raw', line=dict(color='gray')), row=1, col=1)
        # Filtered
        fig_signal.add_trace(go.Scatter(x=time_data[:plot_points], y=filtered_ecg[:plot_points], 
                                        mode='lines', name='Filtered', line=dict(color='blue')), row=2, col=1)
        
        # Peaks (within subset)
        peaks_subset = r_peaks[r_peaks < plot_points]
        fig_signal.add_trace(go.Scatter(x=time_data[peaks_subset], y=filtered_ecg[peaks_subset],
                                        mode='markers', name='R-Peaks', 
                                        marker=dict(color='red', size=8, symbol='star')), row=2, col=1)
                                        
        fig_signal.update_layout(height=600, showlegend=True, title_text="Overall Signal Processing (15s sample)")
        fig_signal.update_xaxes(title_text="Time (s)", row=2, col=1)
        st.plotly_chart(fig_signal, use_container_width=True)

    with tab_pt:
        st.markdown("### Pan-Tompkins Intermediate Signals")
        fig_pt = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                               subplot_titles=("1. Bandpass Filtered (5-15 Hz)", "2. Derivative", "3. Squared", "4. Moving Window Integration"))
        
        fig_pt.add_trace(go.Scatter(x=time_data[:plot_points], y=pt_results['filtered'][:plot_points], 
                                    mode='lines', line=dict(color='blue'), showlegend=False), row=1, col=1)
                                    
        fig_pt.add_trace(go.Scatter(x=time_data[:plot_points], y=pt_results['derivative'][:plot_points], 
                                    mode='lines', line=dict(color='purple'), showlegend=False), row=2, col=1)
                                    
        fig_pt.add_trace(go.Scatter(x=time_data[:plot_points], y=pt_results['squared'][:plot_points], 
                                    mode='lines', line=dict(color='green'), showlegend=False), row=3, col=1)
                                    
        mwi_sub = pt_results['mwi'][:plot_points]
        fig_pt.add_trace(go.Scatter(x=time_data[:plot_points], y=mwi_sub, 
                                    mode='lines', line=dict(color='orange'), name='MWI'), row=4, col=1)
        fig_pt.add_hline(y=pt_results['threshold'], line_dash="dash", line_color="red", annotation_text="Threshold", row=4, col=1)
        
        fig_pt.update_layout(height=800, title_text="Pan-Tompkins Stages (15s sample)")
        fig_pt.update_xaxes(title_text="Time (s)", row=4, col=1)
        st.plotly_chart(fig_pt, use_container_width=True)
    
    # --- HRV DASHBOARD ---
    st.markdown("---")
    st.header("2. Heart Rate Variability (HRV) Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Time-Domain")
        for key, val in td_metrics.items():
            if not np.isnan(val):
                st.metric(label=key, value=f"{val:.2f}")
            else:
                st.metric(label=key, value="N/A")
                
    with col2:
        st.subheader("Frequency-Domain")
        st.metric(label="LF Power", value=f"{fd_metrics.get('LF Power', np.nan):.2f}")
        st.metric(label="HF Power", value=f"{fd_metrics.get('HF Power', np.nan):.2f}")
        st.metric(label="LF/HF Ratio", value=f"{fd_metrics.get('LF/HF Ratio', np.nan):.2f}")
        
    with col3:
        st.subheader("Non-Linear")
        st.metric(label="SD1", value=f"{nl_metrics.get('SD1', np.nan):.2f}")
        st.metric(label="SD2", value=f"{nl_metrics.get('SD2', np.nan):.2f}")
        st.metric(label="Sample Entropy", value=f"{nl_metrics.get('Sample Entropy', np.nan):.4f}")
        
    # HRV Plots
    st.subheader("Analysis Visualizations")
    tab1, tab2, tab3 = st.tabs(["RR Tachogram", "Poincaré Plot", "Power Spectral Density"])
    
    with tab1:
        fig_rr = go.Figure()
        rr_times = r_peaks[1:] / fs
        fig_rr.add_trace(go.Scatter(x=rr_times, y=rr_intervals, mode='lines+markers', line=dict(color='orange')))
        fig_rr.update_layout(title="RR Interval Tachogram", xaxis_title="Time (s)", yaxis_title="RR Interval (ms)")
        st.plotly_chart(fig_rr, use_container_width=True)
        
    with tab2:
        fig_poincare = go.Figure()
        rr_n = nl_metrics.get('rr_n', [])
        rr_n1 = nl_metrics.get('rr_n1', [])
        if len(rr_n) > 0:
            fig_poincare.add_trace(go.Scatter(x=rr_n, y=rr_n1, mode='markers', 
                                              marker=dict(size=6, color='blue', opacity=0.6), name='RR Intervals'))
            # Add identity line
            min_rr = min(min(rr_n), min(rr_n1))
            max_rr = max(max(rr_n), max(rr_n1))
            fig_poincare.add_trace(go.Scatter(x=[min_rr, max_rr], y=[min_rr, max_rr], mode='lines', 
                                              name='Identity', line=dict(color='red', dash='dash')))
                                              
            # Add SD1/SD2 Ellipse
            sd1 = nl_metrics.get('SD1', np.nan)
            sd2 = nl_metrics.get('SD2', np.nan)
            mean_n = nl_metrics.get('mean_rr_n', np.nan)
            mean_n1 = nl_metrics.get('mean_rr_n1', np.nan)
            
            if not np.isnan(sd1) and not np.isnan(sd2) and not np.isnan(mean_n):
                t = np.linspace(0, 2*np.pi, 100)
                # Angle is 45 degrees (pi/4)
                theta = np.pi / 4
                x_ellipse = mean_n + sd2 * np.cos(t) * np.cos(theta) - sd1 * np.sin(t) * np.sin(theta)
                y_ellipse = mean_n1 + sd2 * np.cos(t) * np.sin(theta) + sd1 * np.sin(t) * np.cos(theta)
                
                fig_poincare.add_trace(go.Scatter(x=x_ellipse, y=y_ellipse, mode='lines', 
                                                  name='SD1/SD2 Ellipse', line=dict(color='green', width=2)))

            fig_poincare.update_layout(title="Poincaré Plot", xaxis_title="RR(n) [ms]", yaxis_title="RR(n+1) [ms]", 
                                       showlegend=True, width=600, height=600)
            st.plotly_chart(fig_poincare, use_container_width=True)
        else:
            st.warning("Not enough data for Poincaré plot.")
            
    with tab3:
        freqs = fd_metrics.get('freqs', [])
        psd = fd_metrics.get('psd', [])
        if len(freqs) > 0:
            fig_psd = go.Figure()
            fig_psd.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', fill='tozeroy', line=dict(color='purple')))
            
            # Highlight bands
            fig_psd.add_vrect(x0=0.04, x1=0.15, fillcolor="yellow", opacity=0.2, layer="below", line_width=0, annotation_text="LF")
            fig_psd.add_vrect(x0=0.15, x1=0.40, fillcolor="green", opacity=0.2, layer="below", line_width=0, annotation_text="HF")
            
            fig_psd.update_layout(title="Power Spectral Density (Welch's Method)", 
                                  xaxis_title="Frequency (Hz)", yaxis_title="Density",
                                  xaxis_range=[0, 0.5])
            st.plotly_chart(fig_psd, use_container_width=True)
        else:
            st.warning("Not enough data for PSD plot.")

else:
    st.info("Please upload an ECG `.csv` file. You can run `python data_generator.py` to generate a `sample_ecg.csv` for demonstration.")
