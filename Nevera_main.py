import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os

# Page configuration
st.set_page_config(
    page_title="Nevera - Vibration Source Locator",
    page_icon="📍",
    layout="wide"
)

# Helper functions
def load_and_process_signal(uploaded_file, column_name):
    """Load and process sensor data from CSV"""
    try:
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
            
        return signal_data
        
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None

def butter_bandpass(lowcut, highcut, fs, order=3):
    """Design Butterworth bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut, highcut, fs, order=3):
    """Apply bandpass filter to signal"""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = signal.filtfilt(b, a, data)
    return y

def calculate_rms(data):
    """Calculate RMS amplitude of signal"""
    return np.sqrt(np.mean(data**2))

def get_apollonian_locus(p1, p2, k):
    """Calculate Apollonian circle parameters"""
    x1, y1 = p1
    x2, y2 = p2
    k_sq = k**2
    
    # Handle k=1 case (perpendicular bisector)
    if abs(k - 1.0) < 1e-7:
        A = 2*(x1 - x2)
        B = 2*(y1 - y2)
        C = x2**2 + y2**2 - x1**2 - y1**2
        return {'type': 'line', 'A': A, 'B': B, 'C': C}
    
    # Calculate circle parameters
    denom = 1 - k_sq
    h = (x2 - k_sq*x1) / denom
    kc = (y2 - k_sq*y1) / denom
    
    # Calculate radius
    d_p1p2 = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    r = (k * d_p1p2) / abs(denom)
    
    return {'type': 'circle', 'center': (h, kc), 'radius': r}

def intersect_loci(locus1, locus2):
    """Find intersection points between two loci"""
    if locus1['type'] == 'line' and locus2['type'] == 'line':
        return intersect_two_lines(locus1, locus2)
    elif locus1['type'] == 'circle' and locus2['type'] == 'circle':
        return intersect_two_circles(locus1, locus2)
    elif (locus1['type'] == 'line' and locus2['type'] == 'circle') or \
         (locus1['type'] == 'circle' and locus2['type'] == 'line'):
        line = locus1 if locus1['type'] == 'line' else locus2
        circle = locus2 if locus2['type'] == 'circle' else locus1
        return intersect_line_circle(line, circle)
    return [], []

def intersect_two_lines(line1, line2):
    """Find intersection of two lines"""
    A1, B1, C1 = line1['A'], line1['B'], line1['C']
    A2, B2, C2 = line2['A'], line2['B'], line2['C']
    
    det = A1*B2 - A2*B1
    if abs(det) < 1e-7:
        return []  # Parallel lines
    
    x = (B1*C2 - B2*C1) / det
    y = (A2*C1 - A1*C2) / det
    return [(x, y)]

def intersect_two_circles(circle1, circle2):
    """Find intersection of two circles"""
    h1, k1 = circle1['center']
    r1 = circle1['radius']
    h2, k2 = circle2['center']
    r2 = circle2['radius']
    
    # Distance between centers
    d = np.sqrt((h2-h1)**2 + (k2-k1)**2)
    
    # Check for no solution
    if d > r1 + r2 or d < abs(r1 - r2):
        return []
    
    # Calculate intersection points
    a = (r1**2 - r2**2 + d**2) / (2*d)
    h = a / d
    p = np.sqrt(r1**2 - a**2)
    
    # Midpoint
    xm = h1 + h*(h2 - h1)
    ym = k1 + h*(k2 - k1)
    
    # Intersection points
    xs1 = xm + p*(k2 - k1)/d
    ys1 = ym - p*(h2 - h1)/d
    xs2 = xm - p*(k2 - k1)/d
    ys2 = ym + p*(h2 - h1)/d
    
    if abs(p) < 1e-7:  # Tangent (one solution)
        return [(xs1, ys1)]
    return [(xs1, ys1), (xs2, ys2)]

def intersect_line_circle(line, circle):
    """Find intersection of line and circle"""
    A, B, C = line['A'], line['B'], line['C']
    h, k = circle['center']
    r = circle['radius']
    
    # Handle vertical line
    if abs(B) < 1e-7:
        x = -C/A
        # Solve quadratic for y
        a = 1
        b = -2*k
        c = k**2 - r**2 + (x - h)**2
        disc = b**2 - 4*a*c
        if disc < 0:
            return []
        y1 = (-b + np.sqrt(disc)) / (2*a)
        y2 = (-b - np.sqrt(disc)) / (2*a)
        return [(x, y1), (x, y2)]
    
    # Non-vertical line: y = mx + c
    m = -A/B
    c = -C/B
    
    # Solve quadratic equation
    a_coeff = 1 + m**2
    b_coeff = 2*(m*(c - k) - h)
    c_coeff = h**2 + (c - k)**2 - r**2
    
    disc = b_coeff**2 - 4*a_coeff*c_coeff
    if disc < 0:
        return []
    
    # Calculate x coordinates
    x1 = (-b_coeff + np.sqrt(disc)) / (2*a_coeff)
    x2 = (-b_coeff - np.sqrt(disc)) / (2*a_coeff)
    
    # Calculate y coordinates
    y1 = m*x1 + c
    y2 = m*x2 + c
    
    if disc < 1e-7:  # Tangent (one solution)
        return [(x1, y1)]
    return [(x1, y1), (x2, y2)]

# Main application
def main():
    # Welcome screen
    st.title(" Nevera - Vibration Source Locator")
    st.image("https://images.unsplash.com/photo-1581092580497-e0d23cbdf1dc?auto=format&fit=crop&w=1200&h=400", 
             caption="Vibration Analysis System", use_column_width=True)
    
    st.markdown("""
    ## Welcome to Nevera!
    This application estimates the location of vibration sources using data from three sensors.
    Upload your sensor data and configure the parameters to get started.
    """)
    
    with st.expander("How it works:"):
        st.markdown("""
        1. **Upload CSV files** containing vibration data from three sensors
        2. **Specify sensor locations** in 2D coordinates
        3. **Set parameters**: Sampling rate, frequency filter range, and attenuation exponent
        4. **Process data** to estimate vibration source location
        5. **Visualize results** on an interactive plot
        
        The algorithm uses:
        - Bandpass filtering to isolate relevant frequencies
        - RMS amplitude calculation for signal intensity
        - Geometric loci (Apollonian circles) based on attenuation model
        - Intersection of loci to estimate source location
        """)
    
    st.divider()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Configuration")
        
        # Sensor locations
        st.subheader("Sensor Locations")
        s1_x = st.number_input("Sensor 1 X", value=0.0)
        s1_y = st.number_input("Sensor 1 Y", value=0.0)
        s2_x = st.number_input("Sensor 2 X", value=0.0)
        s2_y = st.number_input("Sensor 2 Y", value=26.5)
        s3_x = st.number_input("Sensor 3 X", value=30.0)
        s3_y = st.number_input("Sensor 3 Y", value=26.5)
        
        # Actual source (optional)
        st.subheader("Actual Source (Optional)")
        actual_x = st.number_input("Actual Source X", value=4.0)
        actual_y = st.number_input("Actual Source Y", value=14.0)
        show_actual = st.checkbox("Show actual source on plot", value=True)
        
        # Parameters
        st.subheader("Processing Parameters")
        column_name = st.text_input("Data Column Name", "acceleration")
        sampling_freq = st.number_input("Sampling Frequency (Hz)", min_value=1.0, value=333.33)
        low_cutoff = st.number_input("Low Cutoff Frequency (Hz)", min_value=0.1, value=1.0)
        high_cutoff = st.number_input("High Cutoff Frequency (Hz)", min_value=1.0, value=22.0)
        attenuation_n = st.number_input("Attenuation Exponent (n)", min_value=0.1, value=1.0, 
                                       help="1 for cylindrical waves, 2 for spherical waves")
    
    # Main content area
    st.header("Upload Sensor Data")
    
    # File uploaders
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Sensor 1 Data")
        sensor1_file = st.file_uploader("Upload CSV for Sensor 1", type="csv", key="s1")
    with col2:
        st.subheader("Sensor 2 Data")
        sensor2_file = st.file_uploader("Upload CSV for Sensor 2", type="csv", key="s2")
    with col3:
        st.subheader("Sensor 3 Data")
        sensor3_file = st.file_uploader("Upload CSV for Sensor 3", type="csv", key="s3")
    
    # Process button
    if st.button("Locate Vibration Source", use_container_width=True):
        if not (sensor1_file and sensor2_file and sensor3_file):
            st.error("Please upload all three sensor data files")
            return
            
        with st.spinner("Processing data..."):
            try:
                # Load and process data
                signal1 = load_and_process_signal(sensor1_file, column_name)
                signal2 = load_and_process_signal(sensor2_file, column_name)
                signal3 = load_and_process_signal(sensor3_file, column_name)
                
                if signal1 is None or signal2 is None or signal3 is None:
                    return
                
                # Apply filters
                filtered1 = apply_filter(signal1, low_cutoff, high_cutoff, sampling_freq)
                filtered2 = apply_filter(signal2, low_cutoff, high_cutoff, sampling_freq)
                filtered3 = apply_filter(signal3, low_cutoff, high_cutoff, sampling_freq)
                
                # Calculate intensities
                i1 = calculate_rms(filtered1)
                i2 = calculate_rms(filtered2)
                i3 = calculate_rms(filtered3)
                
                # Calculate distance ratios
                k21 = (i1 / i2) ** (1/attenuation_n)
                k31 = (i1 / i3) ** (1/attenuation_n)
                
                # Get loci
                locus12 = get_apollonian_locus((s1_x, s1_y), (s2_x, s2_y), k21)
                locus13 = get_apollonian_locus((s1_x, s1_y), (s3_x, s3_y), k31)
                
                # Find intersections
                intersections = intersect_loci(locus12, locus13)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Plot sensor locations
                ax.plot(s1_x, s1_y, 'bo', markersize=10, label='Sensor 1')
                ax.plot(s2_x, s2_y, 'ro', markersize=10, label='Sensor 2')
                ax.plot(s3_x, s3_y, 'go', markersize=10, label='Sensor 3')
                
                # Plot actual source if enabled
                if show_actual:
                    ax.plot(actual_x, actual_y, 'm*', markersize=15, label='Actual Source')
                
                # Plot loci
                plot_locus(ax, locus12, 'S1-S2 Locus', 'blue')
                plot_locus(ax, locus13, 'S1-S3 Locus', 'green')
                
                # Plot intersections
                if intersections:
                    for i, (x, y) in enumerate(intersections):
                        ax.plot(x, y, 'ks', markersize=12, fillstyle='none', 
                                label=f'Estimated Source {i+1}')
                        ax.text(x+0.5, y+0.5, f'({x:.1f}, {y:.1f})', fontsize=9)
                
                # Configure plot
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.set_title('Vibration Source Localization')
                ax.legend(loc='best')
                ax.grid(True)
                ax.axis('equal')
                
                # Display results
                st.success("Processing complete!")
                st.pyplot(fig)
                
                # Display numerical results
                st.subheader("Results")
                st.write(f"- **Sensor 1 Intensity (RMS):** {i1:.4f}")
                st.write(f"- **Sensor 2 Intensity (RMS):** {i2:.4f}")
                st.write(f"- **Sensor 3 Intensity (RMS):** {i3:.4f}")
                st.write(f"- **Distance Ratio S2/S1 (k21):** {k21:.4f}")
                st.write(f"- **Distance Ratio S3/S1 (k31):** {k31:.4f}")
                
                if intersections:
                    for i, (x, y) in enumerate(intersections):
                        st.success(f"Estimated Source Location {i+1}: ({x:.2f}, {y:.2f})")
                        if show_actual:
                            distance = np.sqrt((x - actual_x)**2 + (y - actual_y)**2)
                            st.info(f"Distance from actual source: {distance:.2f} units")
                else:
                    st.warning("No intersection points found. Try adjusting parameters.")
                    
            except Exception as e:
                st.error(f"Processing error: {str(e)}")

def plot_locus(ax, locus, label, color):
    """Plot geometric locus on matplotlib axis"""
    if locus['type'] == 'line':
        # For lines: plot within current axis limits
        xlim = ax.get_xlim()
        if abs(locus['B']) < 1e-7:  # Vertical line
            x = -locus['C']/locus['A']
            ylim = ax.get_ylim()
            ax.plot([x, x], ylim, '--', color=color, label=label)
        else:
            # y = mx + c
            m = -locus['A']/locus['B']
            c = -locus['C']/locus['B']
            x_vals = np.array(xlim)
            y_vals = m*x_vals + c
            ax.plot(x_vals, y_vals, '--', color=color, label=label)
    else:
        # For circles
        circle = plt.Circle(locus['center'], locus['radius'], 
                           color=color, fill=False, linestyle=':', label=label)
        ax.add_patch(circle)

if __name__ == "__main__":
    main()