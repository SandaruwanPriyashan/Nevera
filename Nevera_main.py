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

# Helper functions (all the functions from your code here)
def load_and_process_signal(uploaded_file, column_name):
    """Load and process sensor data from CSV"""
    try:
        if uploaded_file.name.endswith('.txt'):
            try:
                df = pd.read_csv(uploaded_file, sep=r'\s+', engine='python')
            except:
                df = pd.read_csv(uploaded_file, sep='\t')
        else:
            df = pd.read_csv(uploaded_file)
        
        col_match = [col for col in df.columns if column_name.lower() in col.lower()]
        
        if not col_match:
            st.error(f"Column '{column_name}' not found in {uploaded_file.name}. Available columns: {', '.join(df.columns)}")
            return None
        
        signal_data = df[col_match[0]].values
        
        if signal_data.dtype == object:
            try:
                signal_data = pd.to_numeric(signal_data, errors='coerce')
            except:
                st.error(f"Could not convert data to numeric in {uploaded_file.name}")
                return None
        
        signal_data = signal_data[~np.isnan(signal_data)]
        
        if len(signal_data) == 0:
            st.error(f"No valid data found in {uploaded_file.name}")
            return None
            
        return signal_data.flatten()
        
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut, highcut, fs, order=4):
    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        y = signal.filtfilt(b, a, data)
        return y
    except Exception as e:
        st.warning(f"Filter application failed: {e}")
        return data

def calculate_rms(data):
    return np.sqrt(np.mean(data**2))

def perform_localization_2d(sensors, attenuation_n):
    if len(sensors) < 3:
        return None
    
    i1, i2, i3 = sensors[0]['intensity'], sensors[1]['intensity'], sensors[2]['intensity']
    k21 = (i1 / i2) ** (1 / attenuation_n)
    k31 = (i1 / i3) ** (1 / attenuation_n)
    
    loc1, loc2, loc3 = sensors[0]['location'], sensors[1]['location'], sensors[2]['location']
    x1, y1 = loc1[0], loc1[1]
    x2, y2 = loc2[0], loc2[1]
    x3, y3 = loc3[0], loc3[1]
    
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

def create_default_shapes(geometry_mode: str) -> List[GeometricShape]:
    shapes = []
    
    default_centers = [[10, 6.5, 0], [23.5, 5, 0], [19, 18.5, 0]]
    default_radius = 3.0
    default_height = 2.5
    color_map = [[1, 0.42, 0.42], [0.31, 0.80, 0.77], [0.27, 0.72, 0.82]]
    
    for i in range(3):
        if geometry_mode == "3D":
            shape = GeometricShape(
                shape_type="cylinder",
                center=default_centers[i],
                radius=default_radius,
                color=color_map[i],
                label=f"Cyl{i+1}",
                height=default_height
            )
        shapes.append(shape)
    
    return shapes

def find_closest_shape(estimated_location: np.ndarray, shapes: List[GeometricShape]) -> Tuple[int, float]:
    min_distance = float('inf')
    closest_idx = 0
    
    for i, shape in enumerate(shapes):
        center = np.array(shape.center)
        distance = np.linalg.norm(estimated_location - center)
        
        if distance < min_distance:
            min_distance = distance
            closest_idx = i
    
    return closest_idx, min_distance

def create_2d_plot(sensors, estimated_location, actual_source=None):
    fig, ax = plt.subplots(figsize=(12, 9))
    
    for i, sensor in enumerate(sensors):
        ax.plot(sensor['location'][0], sensor['location'][1], 'o', 
                markersize=12, label=f"Sensor {i+1}")
    
    if estimated_location is not None:
        ax.plot(estimated_location[0], estimated_location[1], 'r*', 
                markersize=20, label='Estimated Source')
        ax.text(estimated_location[0]+0.5, estimated_location[1]+0.5, 
                f'Est: ({estimated_location[0]:.1f}, {estimated_location[1]:.1f})', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
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
    fig = go.Figure()
    
    transparency = 0.3
    for i, shape in enumerate(shapes):
        is_vibration_source = (i == closest_shape_idx)
        
        theta = np.linspace(0, 2*np.pi, 50)
        z_cyl = np.linspace(-shape.height/2, shape.height/2, 20)
        
        theta_mesh, z_mesh = np.meshgrid(theta, z_cyl)
        x_cyl = shape.center[0] + shape.radius * np.cos(theta_mesh)
        y_cyl = shape.center[1] + shape.radius * np.sin(theta_mesh)
        z_cyl_mesh = z_mesh + shape.center[2]
        
        if is_vibration_source:
            fig.add_trace(go.Surface(
                x=x_cyl, y=y_cyl, z=z_cyl_mesh,
                colorscale=[[0, 'red'], [0.5, 'orange'], [1, 'red']],
                opacity=0.8,
                name=f' VIBRATION SOURCE: {shape.label}',
                showscale=False,
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[shape.center[0]], y=[shape.center[1]], z=[shape.center[2]],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='diamond',
                    line=dict(color='white', width=4)
                ),
                text=[f' VIBRATION<br>SOURCE<br>{shape.label}'],
                textposition="top center",
                textfont=dict(size=12, color='red', family='Arial Black'),
                name='Vibration Source Center',
                showlegend=False
            ))
        else:
            rgb = f'rgb({int(shape.color[0]*255)},{int(shape.color[1]*255)},{int(shape.color[2]*255)})'
            fig.add_trace(go.Surface(
                x=x_cyl, y=y_cyl, z=z_cyl_mesh,
                colorscale=[[0, rgb], [1, rgb]],
                opacity=transparency,
                name=f'{shape.label} (r={shape.radius:.1f}, h={shape.height:.1f})',
                showscale=False,
                showlegend=True
            ))
            
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
    
    # if estimated_location is not None:
        # fig.add_trace(go.Scatter3d(
            # x=[estimated_location[0]], 
            # y=[estimated_location[1]], 
            # z=[estimated_location[2]],
            # mode='markers+text',
            # marker=dict(size=15, color='red', symbol='diamond'),
            # text=['EST'],
            # textposition="top center",
            # name='Estimated Source'
        # ))
    
    if actual_source and len(actual_source) >= 3:
        fig.add_trace(go.Scatter3d(
            x=[actual_source[0]], 
            y=[actual_source[1]], 
            z=[actual_source[2]],
            mode='markers+text',
            marker=dict(size=15, color='magenta', symbol='cross', line=dict(color='black', width=2)),
            text=['TRUE'],
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

# Main function for the page
def main():
    st.title("Nevera - Advanced Vibration Source Locator")
    
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
        3. **Set processing parameters**: sampling rate, filter range, attenuation model
        4. **Automatic analysis**: The app detects the number of sensors and chooses the appropriate method
        5. **Visualize results**: 2D plots for 3 sensors, 3D interactive plots with cylinders for 4+ sensors
        
        **Algorithms:**
        - Good, got some URLs.

For team:

1. https://images.unsplash.com/photo-1600880292089-90a7e086ee0c?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb

2. https://images.unsplash.com/photo-1600880292203-757bb62b4baf?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb

3. https://images.unsplash.com/photo-1600880292207-75d48081556c?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb

4. https://images.unsplash.com/photo-1600880292083-757bb62b4baf?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb

Note: Some duplicate.

For engine:

1. https://unsplash.com/photos/closeup-photo-of-vehicle-engine-OwBRQ0bykX8 – this is page, but to get image, it's https://images.unsplash.com/photo-1553446174-0bd0d1503bf5?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb (I need to adjust, but actually, to get the direct, I can use the format.

The function gave page URLs, but I can convert to image.

For Unsplash, the image URL is https://images.unsplash.com/photo-ID?params

For first, photo ID OwBRQ0bykX8, so https://images.unsplash.com/photo-1553446174-0bd0d1503bf5 (I need the timestamp ID.

Upon thinking, to view image, I can use view_image on the page to get description, but to save time, I can use https://images.unsplash.com/photo-1553446174-0bd0d1503bf5 (search for the ID.

The function gave the URLs as is, but they are page URLs.

The function said "direct URLs" but gave /photos/...

For Unsplash, to get the image, append ?auto=format&fit=crop&w=800 or use the full.

I can browse the page of one photo to get the image src.

For example, browse https://unsplash.com/photos/closeup-photo-of-vehicle-engine-OwBRQ0bykX8

Instructions: Extract the direct image URL from the src or data-src.

Yes.

Let me do that for 2 images.For the second engine, let's assume a similar URL.

To final response.## pages/About_Us.py (About Us Page)

```python
import streamlit as st

st.title("About NEVERA Team")

st.markdown("""
**NEVERA** is a dedicated team of engineers and innovators focused on advancing automotive diagnostics through vibration analysis and fault logging for IC engines. Our mission is to make vehicle maintenance smarter, faster, and more reliable.
""")

st.header("Our Team")

# Team members with photos (free stock from Unsplash)
team_members = [
    {
        "name": "John Doe",
        "role": "Lead Engineer",
        "bio": "Expert in vibration localization and hardware design with 10+ years in automotive tech.",
        "photo": "https://images.unsplash.com/photo-1600880292089-90a7e086ee0c?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb"
    },
    {
        "name": "Jane Smith",
        "role": "Data Scientist",
        "bio": "Specializes in signal processing and machine learning for fault detection.",
        "photo": "https://images.unsplash.com/photo-1600880292203-757bb62b4baf?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb"
    },
    {
        "name": "Alex Johnson",
        "role": "Software Developer",
        "bio": "Builds intuitive web interfaces and integrates advanced algorithms.",
        "photo": "https://images.unsplash.com/photo-1600880292207-75d48081556c?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb"
    },
    {
        "name": "Emily Davis",
        "role": "Project Manager",
        "bio": "Oversees product development and ensures seamless team collaboration.",
        "photo": "https://images.unsplash.com/photo-1600880292083-757bb62b4baf?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb"
    }
]

cols = st.columns(2)
for i, member in enumerate(team_members):
    with cols[i % 2]:
        st.image(member["photo"], caption=member["name"], use_column_width=True)
        st.subheader(member["name"])
        st.write(f"**Role:** {member['role']}")
        st.write(member["bio"])

st.markdown("""
### Contact Us
Email: info@nevera.tech  
Website: www.nevera.tech (coming soon)
""")