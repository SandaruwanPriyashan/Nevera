import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import signal
from scipy.signal import butter, filtfilt
import warnings
from typing import List, Tuple

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Nevera - Advanced Vibration Source Locator",
    page_icon="📍",
    layout="wide"
)

# Enhanced Classes for Geometric Shapes
class GeometricShape:
    def __init__(self, shape_type: str, center: List[float], radius: float, 
                 color: List[float], label: str, height: float = None):
        self.shape_type = shape_type  # 'circle' or 'cylinder'
        self.center = center
        self.radius = radius
        self.color = color
        self.label = label
        self.height = height  # Only for cylinders

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

def perform_localization_2d(sensors, attenuation_n):
    """Perform 2D localization using 3 sensors"""
    if len(sensors) < 3:
        return None
    
    i1, i2, i3 = sensors[0]['intensity'], sensors[1]['intensity'], sensors[2]['intensity']
    k21 = (i1 / i2) ** (1 / attenuation_n)
    k31 = (i1 / i3) ** (1 / attenuation_n)
    
    loc1, loc2, loc3 = sensors[0]['location'], sensors[1]['location'], sensors[2]['location']
    x1, y1 = loc1[0], loc1[1]
    x2, y2 = loc2[0], loc2[1]
    x3, y3 = loc3[0], loc3[1]
    
    # Matrix formulation for 2D localization
    A = 2 * np.array([
        [x2 - k21**2 * x1, y2 - k21**2 * y1],
        [x3 - k31**2 * x1, y3 - k31**2 * y1]
    ])
    
    b = np.array([
        (x2**2 + y2**2) - k21**2 * (x1**2 + y1**2),
        (x3**2 + y3**2) - k31**2 * (x1**2 + y1**2)
    ])
    
    try:
        if np.linalg.matrix_rank(A) == 2:
            solution = np.linalg.solve(A, b)
            return solution
    except:
        pass
    
    return None

def perform_localization_3d(sensors, attenuation_n):
    """Perform 3D localization using 4+ sensors"""
    if len(sensors) < 4:
        return None
    
    i1, i2, i3, i4 = sensors[0]['intensity'], sensors[1]['intensity'], sensors[2]['intensity'], sensors[3]['intensity']
    k21 = (i1 / i2) ** (1 / attenuation_n)
    k31 = (i1 / i3) ** (1 / attenuation_n)
    k41 = (i1 / i4) ** (1 / attenuation_n)
    
    loc1, loc2, loc3, loc4 = sensors[0]['location'], sensors[1]['location'], sensors[2]['location'], sensors[3]['location']
    x1, y1, z1 = loc1[0], loc1[1], loc1[2]
    x2, y2, z2 = loc2[0], loc2[1], loc2[2]
    x3, y3, z3 = loc3[0], loc3[1], loc3[2]
    x4, y4, z4 = loc4[0], loc4[1], loc4[2]
    
    # Matrix formulation for 3D localization
    A = 2 * np.array([
        [x2 - k21**2 * x1, y2 - k21**2 * y1, z2 - k21**2 * z1],
        [x3 - k31**2 * x1, y3 - k31**2 * y1, z3 - k31**2 * z1],
        [x4 - k41**2 * x1, y4 - k41**2 * y1, z4 - k41**2 * z1]
    ])
    
    b = np.array([
        (x2**2 + y2**2 + z2**2) - k21**2 * (x1**2 + y1**2 + z1**2),
        (x3**2 + y3**2 + z3**2) - k31**2 * (x1**2 + y1**2 + z1**2),
        (x4**2 + y4**2 + z4**2) - k41**2 * (x1**2 + y1**2 + z1**2)
    ])
    
    try:
        if np.linalg.matrix_rank(A) == 3:
            solution = np.linalg.solve(A, b)
            return solution
    except:
        pass
    
    return None

def create_user_defined_shapes(user_shapes_data: List[dict]) -> List[GeometricShape]:
    """Create geometric shapes from user input"""
    shapes = []
    color_palette = [
        [1, 0.42, 0.42],    # Light red
        [0.31, 0.80, 0.77], # Teal
        [0.27, 0.72, 0.82], # Light blue
        [0.85, 0.65, 0.13], # Gold
        [0.56, 0.93, 0.56], # Light green
        [0.93, 0.51, 0.93], # Magenta
        [1.0, 0.65, 0.0],   # Orange
        [0.5, 0.0, 0.5]     # Purple
    ]
    
    for i, shape_data in enumerate(user_shapes_data):
        color = color_palette[i % len(color_palette)]
        shape = GeometricShape(
            shape_type="cylinder",
            center=shape_data['center'],
            radius=shape_data['radius'],
            color=color,
            label=shape_data['label'],
            height=shape_data['height']
        )
        shapes.append(shape)
    
    return shapes

def find_closest_shape(estimated_location: np.ndarray, shapes: List[GeometricShape]) -> Tuple[int, float]:
    """Find the closest geometric shape to estimated location"""
    if not shapes:
        return -1, float('inf')
    
    min_distance = float('inf')
    closest_idx = 0
    
    for i, shape in enumerate(shapes):
        center = np.array(shape.center)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(estimated_location - center)
        
        if distance < min_distance:
            min_distance = distance
            closest_idx = i
    
    return closest_idx, min_distance

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

def create_3d_plot_with_shapes(sensors, estimated_location, shapes, closest_shape_idx, actual_source=None):
    """Create enhanced 3D plot with vibration cylinders"""
    fig = go.Figure()
    
    # Add cylinders with vibration source highlighting
    transparency = 0.3
    for i, shape in enumerate(shapes):
        is_vibration_source = (i == closest_shape_idx and closest_shape_idx >= 0)
        
        # Create cylinder surface
        theta = np.linspace(0, 2*np.pi, 50)
        z_cyl = np.linspace(-shape.height/2, shape.height/2, 20)
        
        theta_mesh, z_mesh = np.meshgrid(theta, z_cyl)
        x_cyl = shape.center[0] + shape.radius * np.cos(theta_mesh)
        y_cyl = shape.center[1] + shape.radius * np.sin(theta_mesh)
        z_cyl_mesh = z_mesh + shape.center[2]
        
        if is_vibration_source:
            # VIBRATION SOURCE - Enhanced styling
            fig.add_trace(go.Surface(
                x=x_cyl, y=y_cyl, z=z_cyl_mesh,
                colorscale=[[0, 'red'], [0.5, 'orange'], [1, 'red']],
                opacity=0.8,
                name=f'🔥 VIBRATION SOURCE: {shape.label}',
                showscale=False,
                showlegend=True
            ))
            
            # Vibration source center
            fig.add_trace(go.Scatter3d(
                x=[shape.center[0]], y=[shape.center[1]], z=[shape.center[2]],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='diamond',
                    line=dict(color='white', width=4)
                ),
                text=[f'🔥 VIBRATION<br>SOURCE<br>{shape.label}'],
                textposition="top center",
                textfont=dict(size=12, color='red', family='Arial Black'),
                name='Vibration Source Center',
                showlegend=False
            ))
        else:
            # Regular cylinders
            rgb = f'rgb({int(shape.color[0]*255)},{int(shape.color[1]*255)},{int(shape.color[2]*255)})'
            fig.add_trace(go.Surface(
                x=x_cyl, y=y_cyl, z=z_cyl_mesh,
                colorscale=[[0, rgb], [1, rgb]],
                opacity=transparency,
                name=f'{shape.label} (r={shape.radius:.1f}, h={shape.height:.1f})',
                showscale=False,
                showlegend=True
            ))
            
            # Center point
            fig.add_trace(go.Scatter3d(
                x=[shape.center[0]], y=[shape.center[1]], z=[shape.center[2]],
                mode='markers+text',
                marker=dict(
                    size=6,
                    color=rgb,
                    line=dict(color='black', width=1)
                ),
                text=[shape.label],
                textposition="top center",
                name=f'{shape.label} Center',
                showlegend=False
            ))
    
    # Plot sensors
    sensor_colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, sensor in enumerate(sensors):
        color = sensor_colors[i % len(sensor_colors)]
        fig.add_trace(go.Scatter3d(
            x=[sensor['location'][0]], 
            y=[sensor['location'][1]], 
            z=[sensor['location'][2]],
            mode='markers+text',
            marker=dict(size=10, color=color),
            text=[f'S{i+1}'],
            textposition="top center",
            name=f'Sensor {i+1}'
        ))
    
    # Plot estimated location
    if estimated_location is not None:
        fig.add_trace(go.Scatter3d(
            x=[estimated_location[0]], 
            y=[estimated_location[1]], 
            z=[estimated_location[2]],
            mode='markers+text',
            marker=dict(size=15, color='yellow', symbol='diamond', line=dict(color='black', width=2)),
            text=['⭐ EST'],
            textposition="top center",
            name='Estimated Source'
        ))
    
    # Plot actual source if provided
    if actual_source and len(actual_source) >= 3:
        fig.add_trace(go.Scatter3d(
            x=[actual_source[0]], 
            y=[actual_source[1]], 
            z=[actual_source[2]],
            mode='markers+text',
            marker=dict(size=15, color='magenta', symbol='cross', line=dict(color='black', width=2)),
            text=['✓ TRUE'],
            textposition="top center",
            name='Actual Source'
        ))
    
    fig.update_layout(
        title="3D Vibration Source Localization with Cylinders",
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
    st.title("🎯 Nevera - Advanced Vibration Source Locator")
    
    st.markdown("""
    ## Welcome to Advanced Nevera!
    This application performs vibration source localization using multiple sensors with automatic dimension detection:
    - **3 Sensors**: 2D localization with matplotlib visualization
    - **4+ Sensors**: 3D localization with interactive Plotly visualization including vibration cylinders
    """)
    
    with st.expander("How it works:"):
        st.markdown("""
        1. **Upload sensor data files** (in CSV format)
        2. **Configure sensor locations** in 2D or 3D coordinates
        3. **Define geometric shapes** (cylinders) for vibration source identification
        4. **Set processing parameters**: sampling rate, filter range, attenuation model
        5. **Automatic analysis**: The app detects the number of sensors and chooses the appropriate method
        6. **Visualize results**: 2D plots for 3 sensors, 3D interactive plots with cylinders for 4+ sensors
        
        **Algorithms:**
        - Butterworth bandpass filtering for signal preprocessing
        - RMS amplitude calculation for intensity measurement
        - Matrix-based trilateration for multi-dimensional localization
        - Automatic dimension detection based on sensor count
        - Dynamic cylinder visualization for 3D mode
        """)
    
    st.divider()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Signal processing parameters
        st.subheader("Signal Processing")
        column_name = st.text_input("Data Column Name", "")
        sampling_freq = st.number_input("Sampling Frequency (Hz)", min_value=1.0, value=None)
        low_cutoff = st.number_input("Low Cutoff (Hz)", min_value=0.1, value=None)
        high_cutoff = st.number_input("High Cutoff (Hz)", min_value=1.0, value=None)
        filter_order = st.number_input("Filter Order", min_value=1, max_value=10, value=None)
        attenuation_n = st.number_input("Attenuation Exponent", min_value=0.1, value=None)
        
        # Geometric shapes configuration
        st.divider()
        st.subheader("🔺 Geometric Shapes")
        num_shapes = st.number_input("Number of Cylinders", min_value=0, max_value=10, value=0)
        
        # Actual source location
        st.divider()
        st.subheader("📍 Known Source (Optional)")
        actual_x = st.number_input("Actual Source X", value=None)
        actual_y = st.number_input("Actual Source Y", value=None)
        actual_z = st.number_input("Actual Source Z", value=None)
        show_actual = st.checkbox("Show actual source", value=False)
    
    # Geometric shapes input
    user_shapes_data = []
    if num_shapes > 0:
        st.header("🔺 Define Geometric Shapes")
        st.markdown("Configure the cylinders for vibration source identification:")
        
        cols = st.columns(min(3, num_shapes))
        for i in range(num_shapes):
            col_idx = i % 3
            with cols[col_idx]:
                st.subheader(f"Cylinder {i+1}")
                label = st.text_input(f"Label", f"Cyl{i+1}", key=f"shape_label_{i}")
                center_x = st.number_input(f"Center X", value=None, key=f"shape_cx_{i}")
                center_y = st.number_input(f"Center Y", value=None, key=f"shape_cy_{i}")
                center_z = st.number_input(f"Center Z", value=None, key=f"shape_cz_{i}")
                radius = st.number_input(f"Radius", min_value=0.1, value=None, key=f"shape_r_{i}")
                height = st.number_input(f"Height", min_value=0.1, value=None, key=f"shape_h_{i}")
                
                if all(v is not None for v in [center_x, center_y, center_z, radius, height]):
                    user_shapes_data.append({
                        'label': label,
                        'center': [center_x, center_y, center_z],
                        'radius': radius,
                        'height': height
                    })
    
    # File upload section
    st.header("📁 Upload Sensor Data")
    
    # Create columns for sensor uploads
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        st.subheader("Sensor 1")
        sensor1_file = st.file_uploader("Upload Sensor 1", type=["csv", "txt"], key="s1")
        s1_x = st.number_input("S1 X", value=None, key="s1x")
        s1_y = st.number_input("S1 Y", value=None, key="s1y")
        s1_z = st.number_input("S1 Z", value=None, key="s1z")
    
    with col2:
        st.subheader("Sensor 2")
        sensor2_file = st.file_uploader("Upload Sensor 2", type=["csv", "txt"], key="s2")
        s2_x = st.number_input("S2 X", value=None, key="s2x")
        s2_y = st.number_input("S2 Y", value=None, key="s2y")
        s2_z = st.number_input("S2 Z", value=None, key="s2z")
    
    with col3:
        st.subheader("Sensor 3")
        sensor3_file = st.file_uploader("Upload Sensor 3", type=["csv", "txt"], key="s3")
        s3_x = st.number_input("S3 X", value=None, key="s3x")
        s3_y = st.number_input("S3 Y", value=None, key="s3y")
        s3_z = st.number_input("S3 Z", value=None, key="s3z")
    
    with col4:
        st.subheader("Sensor 4 (Optional)")
        sensor4_file = st.file_uploader("Upload Sensor 4", type=["csv", "txt"], key="s4")
        s4_x = st.number_input("S4 X", value=None, key="s4x")
        s4_y = st.number_input("S4 Y", value=None, key="s4y")
        s4_z = st.number_input("S4 Z", value=None, key="s4z")
    
    # Validation before processing
    def validate_inputs():
        errors = []
        
        # Check required parameters
        if not column_name:
            errors.append("Data Column Name is required")
        if sampling_freq is None:
            errors.append("Sampling Frequency is required")
        if low_cutoff is None:
            errors.append("Low Cutoff frequency is required")
        if high_cutoff is None:
            errors.append("High Cutoff frequency is required")
        if filter_order is None:
            errors.append("Filter Order is required")
        if attenuation_n is None:
            errors.append("Attenuation Exponent is required")
        
        # Check minimum sensor requirements
        uploaded_files = [sensor1_file, sensor2_file, sensor3_file, sensor4_file]
        sensor_locations = [
            [s1_x, s1_y, s1_z],
            [s2_x, s2_y, s2_z],
            [s3_x, s3_y, s3_z],
            [s4_x, s4_y, s4_z]
        ]
        
        valid_sensors = 0
        for i, (file, location) in enumerate(zip(uploaded_files, sensor_locations)):
            if file is not None:
                if any(coord is None for coord in location):
                    errors.append(f"Sensor {i+1} coordinates are incomplete")
                else:
                    valid_sensors += 1
        
        if valid_sensors < 3:
            errors.append("At least 3 sensors with complete data and coordinates are required")
        
        return errors
    
    # Process button
    if st.button("🚀 Perform Advanced Localization", use_container_width=True, type="primary"):
        # Validate inputs
        validation_errors = validate_inputs()
        if validation_errors:
            st.error("❌ Please fix the following issues:")
            for error in validation_errors:
                st.write(f"• {error}")
            return
        
        # Process uploaded sensors
        uploaded_files = [sensor1_file, sensor2_file, sensor3_file, sensor4_file]
        sensor_locations = [
            [s1_x, s1_y, s1_z],
            [s2_x, s2_y, s2_z],
            [s3_x, s3_y, s3_z],
            [s4_x, s4_y, s4_z]
        ]
        
        sensors = []
        for i, (file, location) in enumerate(zip(uploaded_files, sensor_locations)):
            if file is not None and all(coord is not None for coord in location):
                signal_data = load_and_process_signal(file, column_name)
                if signal_data is not None:
                    # Apply filtering
                    filtered_signal = apply_filter(signal_data, low_cutoff, high_cutoff, sampling_freq, filter_order)
                    intensity = calculate_rms(filtered_signal)
                    
                    sensors.append({
                        'location': location,
                        'intensity': intensity,
                        'signal': signal_data,
                        'label': f"S{i+1}",
                        'file': file.name
                    })
        
        num_sensors = len(sensors)
        st.success(f"✅ {num_sensors} sensors detected and processed!")
        
        # Create sensor summary table
        sensor_data = []
        for sensor in sensors:
            sensor_data.append({
                'Sensor': sensor['label'],
                'File': sensor['file'],
                'X': sensor['location'][0],
                'Y': sensor['location'][1],
                'Z': sensor['location'][2],
                'RMS Intensity': f"{sensor['intensity']:.6f}",
                'Samples': len(sensor['signal'])
            })
        
        st.subheader("📊 Sensor Summary")
        st.dataframe(pd.DataFrame(sensor_data), use_container_width=True)
        
        # Perform localization based on sensor count
        with st.spinner("🔍 Performing localization..."):
            if num_sensors == 3:
                st.info("**2D Localization Mode** - Using 3 sensors")
                estimated_location = perform_localization_2d(sensors, attenuation_n)
                localization_mode = "2D"
            else:
                st.info("**3D Localization Mode** - Using 4+ sensors")
                estimated_location = perform_localization_3d(sensors, attenuation_n)
                localization_mode = "3D"
        
        if estimated_location is None:
            st.error("❌ Localization failed. Try adjusting parameters or sensor positions.")
            return
        
        # Display results
        st.subheader("🎯 Localization Results")
        
        # Results metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mode", localization_mode)
            st.metric("Sensors Used", num_sensors)
        
        with col2:
            st.metric("Estimated X", f"{estimated_location[0]:.2f}")
            st.metric("Estimated Y", f"{estimated_location[1]:.2f}")
        
        with col3:
            if len(estimated_location) > 2:
                st.metric("Estimated Z", f"{estimated_location[2]:.2f}")
            else:
                st.metric("Estimated Z", "N/A (2D mode)")
            
            # Calculate error if actual source is provided
            if show_actual and all(v is not None for v in [actual_x, actual_y, actual_z]):
                if localization_mode == "2D":
                    actual = [actual_x, actual_y]
                    error = np.linalg.norm(estimated_location - np.array(actual))
                else:
                    actual = [actual_x, actual_y, actual_z]
                    error = np.linalg.norm(estimated_location - np.array(actual))
                st.metric("Error from True Source", f"{error:.2f}")
        
        # Create visualization
        st.subheader("📈 Visualization")
        
        actual_source = [actual_x, actual_y, actual_z] if (show_actual and all(v is not None for v in [actual_x, actual_y, actual_z])) else None
        
        if localization_mode == "2D":
            fig = create_2d_plot(sensors, estimated_location, actual_source)
            st.pyplot(fig)
        else:
            # Create shapes from user input
            shapes = create_user_defined_shapes(user_shapes_data)
            
            if shapes:
                closest_shape_idx, min_distance = find_closest_shape(estimated_location, shapes)
                fig = create_3d_plot_with_shapes(sensors, estimated_location, shapes, closest_shape_idx, actual_source)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display identified cylinder
                st.subheader("🔥 Identified Vibration Cylinder")
                if closest_shape_idx >= 0:
                    identified_shape = shapes[closest_shape_idx]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Label:** {identified_shape.label}")
                        st.write(f"**Center:** ({identified_shape.center[0]:.1f}, {identified_shape.center[1]:.1f}, {identified_shape.center[2]:.1f})")
                    with col2:
                        st.write(f"**Radius:** {identified_shape.radius:.1f}")
                        st.write(f"**Height:** {identified_shape.height:.1f}")
                        st.write(f"**Distance to Estimated Location:** {min_distance:.2f}")
                else:
                    st.info("No cylinders defined for comparison.")
            else:
                # Show 3D plot without shapes
                fig = go.Figure()
                
                # Plot sensors
                sensor_colors = ['blue', 'red', 'green', 'orange', 'purple']
                for i, sensor in enumerate(sensors):
                    color = sensor_colors[i % len(sensor_colors)]
                    fig.add_trace(go.Scatter3d(
                        x=[sensor['location'][0]], 
                        y=[sensor['location'][1]], 
                        z=[sensor['location'][2]],
                        mode='markers+text',
                        marker=dict(size=10, color=color),
                        text=[f'S{i+1}'],
                        textposition="top center",
                        name=f'Sensor {i+1}'
                    ))
                
                # Plot estimated location
                if estimated_location is not None:
                    fig.add_trace(go.Scatter3d(
                        x=[estimated_location[0]], 
                        y=[estimated_location[1]], 
                        z=[estimated_location[2]],
                        mode='markers+text',
                        marker=dict(size=15, color='yellow', symbol='diamond', line=dict(color='black', width=2)),
                        text=['⭐ EST'],
                        textposition="top center",
                        name='Estimated Source'
                    ))
                
                # Plot actual source if provided
                if actual_source and len(actual_source) >= 3:
                    fig.add_trace(go.Scatter3d(
                        x=[actual_source[0]], 
                        y=[actual_source[1]], 
                        z=[actual_source[2]],
                        mode='markers+text',
                        marker=dict(size=15, color='magenta', symbol='cross', line=dict(color='black', width=2)),
                        text=['✓ TRUE'],
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
                
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 Add geometric shapes above to identify potential vibration sources.")
        
        # Additional analysis
        with st.expander("🔍 Detailed Analysis"):
            st.subheader("Intensity Ratios")
            if num_sensors >= 2:
                k21 = (sensors[0]['intensity'] / sensors[1]['intensity']) ** (1/attenuation_n)
                st.write(f"**k21 (S1/S2):** {k21:.4f}")
            if num_sensors >= 3:
                k31 = (sensors[0]['intensity'] / sensors[2]['intensity']) ** (1/attenuation_n)
                st.write(f"**k31 (S1/S3):** {k31:.4f}")
            if num_sensors >= 4:
                k41 = (sensors[0]['intensity'] / sensors[3]['intensity']) ** (1/attenuation_n)
                st.write(f"**k41 (S1/S4):** {k41:.4f}")
            
            st.subheader("Processing Parameters")
            st.write(f"**Filter Range:** {low_cutoff} - {high_cutoff} Hz")
            st.write(f"**Filter Order:** {filter_order}")
            st.write(f"**Sampling Frequency:** {sampling_freq} Hz")
            st.write(f"**Attenuation Exponent:** {attenuation_n}")
            
            st.subheader("Shape Configuration")
            if user_shapes_data:
                for i, shape_data in enumerate(user_shapes_data):
                    st.write(f"**{shape_data['label']}:** Center=({shape_data['center'][0]}, {shape_data['center'][1]}, {shape_data['center'][2]}), R={shape_data['radius']}, H={shape_data['height']}")
            else:
                st.write("No geometric shapes defined.")

if __name__ == "__main__":
    main()