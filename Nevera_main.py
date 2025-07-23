import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import signal
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Nevera - Advanced Vibration Source Locator",
    page_icon="📍",
    layout="wide"
)

# Helper functions
def load_and_process_signal(uploaded_file, column_name):
    """Load and process sensor data from CSV"""
    try:
        # Handle different file formats
        if uploaded_file.name.endswith('.txt'):
            try:
                df = pd.read_csv(uploaded_file, sep=r'\s+', engine='python')
            except:
                df = pd.read_csv(uploaded_file, sep='\t')
        else:
            df = pd.read_csv(uploaded_file)
        
        # Case-insensitive column matching
        col_match = [col for col in df.columns if column_name.lower() in col.lower()]
        
        if not col_match:
            st.error(f"Column '{column_name}' not found in {uploaded_file.name}. Available columns: {', '.join(df.columns)}")
            return None
        
        signal_data = df[col_match[0]].values
        
        # Convert to numeric if needed
        if signal_data.dtype == object:
            try:
                signal_data = pd.to_numeric(signal_data, errors='coerce')
            except:
                st.error(f"Could not convert data to numeric in {uploaded_file.name}")
                return None
        
        # Remove NaNs
        signal_data = signal_data[~np.isnan(signal_data)]
        
        if len(signal_data) == 0:
            st.error(f"No valid data found in {uploaded_file.name}")
            return None
            
        return signal_data.flatten()
        
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design Butterworth bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut, highcut, fs, order=4):
    """Apply bandpass filter to signal"""
    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        y = signal.filtfilt(b, a, data)
        return y
    except Exception as e:
        st.warning(f"Filter application failed: {e}")
        return data

def calculate_rms(data):
    """Calculate RMS amplitude of signal"""
    return np.sqrt(np.mean(data**2))

def perform_localization_linear(sensors, attenuation_n, num_eq):
    """Perform linear approximation localization using first num_eq+1 sensors"""
    if len(sensors) < num_eq + 1:
        return None
    
    i1 = sensors[0]['intensity']
    loc1 = np.array(sensors[0]['location'])
    
    A = []
    b = []
    for sens in sensors[1: num_eq + 1]:
        ii = sens['intensity']
        loci = np.array(sens['location'])
        k = (i1 / ii) ** (1 / attenuation_n)
        k2 = k ** 2
        
        row = 2 * (loci - k2 * loc1)
        const = np.dot(loci, loci) - k2 * np.dot(loc1, loc1)
        
        A.append(row)
        b.append(const)
    
    A = np.array(A)
    b = np.array(b)
    
    try:
        if np.linalg.matrix_rank(A) == num_eq:
            solution = np.linalg.solve(A, b)
            return solution
    except:
        pass
    
    return None

def perform_localization(sensors, attenuation_n):
    """Perform localization using nonlinear least squares with linear initial guess"""
    if len(sensors) < 3:
        return None
    
    num_sensors = len(sensors)
    dim = 3 if num_sensors >= 4 else 2
    
    # Get initial guess from linear method
    num_eq = dim
    initial_guess = perform_localization_linear(sensors, attenuation_n, num_eq)
    
    if initial_guess is None or len(initial_guess) != dim:
        # Fallback to mean position
        initial_guess = np.mean([np.array(s['location'])[:dim] for s in sensors], axis=0)
    
    # Reference sensor
    loc1 = np.array(sensors[0]['location'])[:dim]
    I1 = sensors[0]['intensity']
    
    def residuals(S):
        S = np.array(S)
        r1 = np.linalg.norm(S - loc1)
        res = []
        for sensor in sensors[1:]:
            loc_i = np.array(sensor['location'])[:dim]
            I_i = sensor['intensity']
            k_i = (I1 / I_i) ** (1 / attenuation_n)
            r_i = np.linalg.norm(S - loc_i)
            res.append(r_i - k_i * r1)
        return res
    
    try:
        result = least_squares(residuals, initial_guess, method='lm')
        if result.success:
            return result.x
        else:
            st.warning("Nonlinear optimization did not converge. Using linear approximation.")
            return initial_guess
    except Exception as e:
        st.warning(f"Optimization failed: {e}. Using linear approximation.")
        return initial_guess

def create_2d_plot(sensors, estimated_location, actual_source=None):
    """Create 2D matplotlib plot"""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot sensors
    for i, sensor in enumerate(sensors):
        ax.plot(sensor['location'][0], sensor['location'][1], 'o', 
                markersize=12, label=f"Sensor {i+1}")
    
    # Plot estimated location
    if estimated_location is not None:
        ax.plot(estimated_location[0], estimated_location[1], 'r*', 
                markersize=20, label='Estimated Source')
        ax.text(estimated_location[0]+0.5, estimated_location[1]+0.5, 
                f'Est: ({estimated_location[0]:.1f}, {estimated_location[1]:.1f})', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot actual source if provided
    if actual_source:
        ax.plot(actual_source[0], actual_source[1], 'm*', 
                markersize=20, label='Actual Source')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('2D Vibration Source Localization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    return fig

def create_3d_plot(sensors, estimated_location, actual_source=None):
    """Create 3D Plotly plot"""
    fig = go.Figure()
    
    # Plot sensors
    sensor_colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, sensor in enumerate(sensors):
        color = sensor_colors[i % len(sensor_colors)]
        loc = sensor['location']
        fig.add_trace(go.Scatter3d(
            x=[loc[0]], 
            y=[loc[1]], 
            z=[loc[2] if len(loc) > 2 else 0],
            mode='markers+text',
            marker=dict(size=10, color=color),
            text=[f'S{i+1}'],
            textposition="top center",
            name=f'Sensor {i+1}'
        ))
    
    # Plot estimated location
    if estimated_location is not None:
        z_est = estimated_location[2] if len(estimated_location) > 2 else 0
        fig.add_trace(go.Scatter3d(
            x=[estimated_location[0]], 
            y=[estimated_location[1]], 
            z=[z_est],
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='diamond'),
            text=['EST'],
            textposition="top center",
            name='Estimated Source'
        ))
    
    # Plot actual source if provided
    if actual_source:
        z_act = actual_source[2] if len(actual_source) > 2 else 0
        fig.add_trace(go.Scatter3d(
            x=[actual_source[0]], 
            y=[actual_source[1]], 
            z=[z_act],
            mode='markers+text',
            marker=dict(size=15, color='magenta', symbol='star'),
            text=['TRUE'],
            textposition="top center",
            name='Actual Source'
        ))
    
    fig.update_layout(
        title="3D Vibration Source Localization",
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Z Position"
        ),
        width=1000,
        height=700
    )
    
    return fig

# Main application
def main():
    # Title and description
    st.title("📍 Nevera - Advanced Vibration Source Locator")
    
    st.markdown("""
    ## Welcome to Advanced Nevera!
    This application performs vibration source localization using multiple sensors with automatic dimension detection:
    - **3+ Sensors**: Unified nonlinear least squares method with linear initial guess
    - Supports 2D and 3D based on sensor coordinates and count
    - Uses all available sensors for better accuracy
    """)
    
    with st.expander("📖 How it works:"):
        st.markdown("""
        1. **Upload sensor data files** (in CSV format)
        2. **Configure sensor locations** in 2D or 3D coordinates
        3. **Set processing parameters**: sampling rate, filter range, attenuation model
        4. **Automatic analysis**: Uses linear approximation as initial guess, refines with nonlinear optimization
        5. **Visualize results**: 2D plots for planar setups, 3D interactive plots otherwise
        
        **Algorithms:**
        - Butterworth bandpass filtering for signal preprocessing
        - RMS amplitude calculation for intensity measurement
        - Nonlinear least squares trilateration based on intensity ratios
        - Handles arbitrary number of sensors (>=3)
        """)
    
    st.divider()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Signal processing parameters
        st.subheader("🔬 Signal Processing")
        column_name = st.text_input("Data Column Name", "accel_y_g")
        sampling_freq = st.number_input("Sampling Frequency (Hz)", min_value=1.0, value=333.33)
        low_cutoff = st.number_input("Low Cutoff (Hz)", min_value=0.1, value=10.0)
        high_cutoff = st.number_input("High Cutoff (Hz)", min_value=1.0, value=50.0)
        filter_order = st.number_input("Filter Order", min_value=1, max_value=10, value=4)
        attenuation_n = st.number_input("Attenuation Exponent", min_value=0.1, value=2.0)
        
        # Actual source location
        st.subheader("🎯 Known Source (Optional)")
        actual_x = st.number_input("Actual Source X", value=23.5)
        actual_y = st.number_input("Actual Source Y", value=5.0)
        actual_z = st.number_input("Actual Source Z", value=1.0)
        show_actual = st.checkbox("Show actual source", value=True)
    
    # File upload section
    st.header("📁 Upload Sensor Data")
    
    # Create columns for sensor uploads
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        st.subheader("📡 Sensor 1")
        sensor1_file = st.file_uploader("Upload Sensor 1", type=["csv", "txt"], key="s1")
        s1_x = st.number_input("S1 X", value=0.0, key="s1x")
        s1_y = st.number_input("S1 Y", value=0.0, key="s1y")
        s1_z = st.number_input("S1 Z", value=0.0, key="s1z")
    
    with col2:
        st.subheader("📡 Sensor 2")
        sensor2_file = st.file_uploader("Upload Sensor 2", type=["csv", "txt"], key="s2")
        s2_x = st.number_input("S2 X", value=20.0, key="s2x")
        s2_y = st.number_input("S2 Y", value=24.5, key="s2y")
        s2_z = st.number_input("S2 Z", value=0.0, key="s2z")
    
    with col3:
        st.subheader("📡 Sensor 3")
        sensor3_file = st.file_uploader("Upload Sensor 3", type=["csv", "txt"], key="s3")
        s3_x = st.number_input("S3 X", value=38.0, key="s3x")
        s3_y = st.number_input("S3 Y", value=0.0, key="s3y")
        s3_z = st.number_input("S3 Z", value=0.0, key="s3z")
    
    with col4:
        st.subheader("📡 Sensor 4 (Optional)")
        sensor4_file = st.file_uploader("Upload Sensor 4", type=["csv", "txt"], key="s4")
        s4_x = st.number_input("S4 X", value=20.0, key="s4x")
        s4_y = st.number_input("S4 Y", value=13.5, key="s4y")
        s4_z = st.number_input("S4 Z", value=-10.0, key="s4z")
    
    # Process button
    if st.button("🚀 Perform Advanced Localization", use_container_width=True, type="primary"):
        # Check minimum requirements
        uploaded_files = [sensor1_file, sensor2_file, sensor3_file, sensor4_file]
        sensor_locations = [
            [s1_x, s1_y, s1_z],
            [s2_x, s2_y, s2_z],
            [s3_x, s3_y, s3_z],
            [s4_x, s4_y, s4_z]
        ]
        
        # Process uploaded sensors
        sensors = []
        for i, (file, location) in enumerate(zip(uploaded_files, sensor_locations)):
            if file is not None:
                signal_data = load_and_process_signal(file, column_name)
                if signal_data is not None:
                    # Apply filtering
                    filtered_signal = apply_filter(signal_data, low_cutoff, high_cutoff, sampling_freq, filter_order)
                    intensity = calculate_rms(filtered_signal)
                    
                    if intensity <= 0:
                        st.warning(f"Non-positive intensity for Sensor {i+1}. Skipping.")
                        continue
                    
                    sensors.append({
                        'location': location,
                        'intensity': intensity,
                        'signal': signal_data,
                        'label': f"S{i+1}",
                        'file': file.name
                    })
        
        num_sensors = len(sensors)
        
        if num_sensors < 3:
            st.error("❌ At least 3 sensors are required for localization!")
            return
        
        # Display sensor information
        st.success(f"✅ {num_sensors} sensors detected!")
        
        # Create sensor summary table
        sensor_data = []
        for sensor in sensors:
            loc = sensor['location']
            sensor_data.append({
                'Sensor': sensor['label'],
                'File': sensor['file'],
                'X': loc[0],
                'Y': loc[1],
                'Z': loc[2] if len(loc) > 2 else 0,
                'RMS Intensity': f"{sensor['intensity']:.6f}",
                'Samples': len(sensor['signal'])
            })
        
        st.subheader("📊 Sensor Summary")
        st.dataframe(pd.DataFrame(sensor_data), use_container_width=True)
        
        # Perform localization
        with st.spinner("🔍 Performing localization..."):
            estimated_location = perform_localization(sensors, attenuation_n)
        
        if estimated_location is None:
            st.error("❌ Localization failed. Try adjusting parameters or sensor positions.")
            return
        
        # Determine dimension
        all_z_zero = all(np.isclose(s['location'][2], 0) for s in sensors if len(s['location']) > 2)
        localization_mode = "2D" if num_sensors == 3 or all_z_zero else "3D"
        if localization_mode == "2D" and len(estimated_location) > 2:
            estimated_location = estimated_location[:2]
        
        # Display results
        st.subheader("📈 Localization Results")
        
        # Results metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mode", localization_mode)
            st.metric("Sensors Used", num_sensors)
        
        with col2:
            st.metric("Estimated X", f"{estimated_location[0]:.2f}")
            st.metric("Estimated Y", f"{estimated_location[1]:.2f}")
        
        with col3:
            if localization_mode == "3D" and len(estimated_location) > 2:
                st.metric("Estimated Z", f"{estimated_location[2]:.2f}")
            else:
                st.metric("Estimated Z", "N/A")
            
            # Calculate error if actual source is provided
            if show_actual:
                if localization_mode == "2D":
                    actual = np.array([actual_x, actual_y])
                else:
                    actual = np.array([actual_x, actual_y, actual_z])
                error = np.linalg.norm(estimated_location - actual)
                st.metric("Error from True Source", f"{error:.2f}")
        
        # Create visualization
        st.subheader("🗺️ Visualization")
        
        actual_source = [actual_x, actual_y, actual_z] if show_actual else None
        
        if localization_mode == "2D":
            fig = create_2d_plot(sensors, estimated_location, actual_source[:2] if actual_source else None)
            st.pyplot(fig)
        else:
            fig = create_3d_plot(sensors, estimated_location, actual_source)
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional analysis
        with st.expander("🔬 Detailed Analysis"):
            st.subheader("Intensity Ratios")
            I1 = sensors[0]['intensity']
            for i, sensor in enumerate(sensors[1:], 1):
                k = (I1 / sensor['intensity']) ** (1/attenuation_n)
                st.write(f"**k{i+1}1 (S1/S{i+1}):** {k:.4f}")
            
            st.subheader("Processing Parameters")
            st.write(f"**Filter Range:** {low_cutoff} - {high_cutoff} Hz")
            st.write(f"**Filter Order:** {filter_order}")
            st.write(f"**Sampling Frequency:** {sampling_freq} Hz")
            st.write(f"**Attenuation Exponent:** {attenuation_n}")

if __name__ == "__main__":
    main()