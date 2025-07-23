# 🎯 Complete Guide: How to Use 3D Vibration Visualization in Streamlit
# Copy this code into your Streamlit application

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal
from scipy.signal import butter, filtfilt

# ===== STEP 1: ADD THESE FUNCTIONS TO YOUR EXISTING CODE =====

def create_clean_3d_visualization(closest_shape_idx, num_sensors):
    """
    Creates clean 3D visualization showing only cylinders/circles with vibration source highlighted
    closest_shape_idx: Index of the shape closest to estimated location (0, 1, or 2)
    num_sensors: Number of sensors (3 = 2D circles, 4+ = 3D cylinders)
    """
    
    # Default shape locations (same as your MATLAB code)
    default_centers = [[10, 6.5, 0], [23.5, 5, 0], [19, 18.5, 0]]
    default_radius = 3.0
    default_height = 2.5
    shape_colors = [[33, 150, 243], [255, 0, 0], [76, 175, 80]]  # Blue, Red, Green
    shape_labels = ['Shape1', 'Shape2', 'Shape3']
    
    # Determine visualization mode
    if num_sensors == 3:
        return create_2d_clean_plot(default_centers, closest_shape_idx, shape_colors, shape_labels)
    else:
        return create_3d_clean_plot(default_centers, closest_shape_idx, shape_colors, shape_labels, default_radius, default_height)

def create_2d_clean_plot(centers, closest_idx, colors, labels):
    """Create 2D circle plot with vibration source highlighting"""
    fig = go.Figure()
    
    for i, (center, color, label) in enumerate(zip(centers, colors, labels)):
        # Generate circle coordinates
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = center[0] + 3.0 * np.cos(theta)  # radius = 3.0
        y_circle = center[1] + 3.0 * np.sin(theta)
        
        is_vibration_source = (i == closest_idx)
        
        # Circle outline and fill
        fig.add_trace(go.Scatter(
            x=x_circle, y=y_circle,
            fill='toself',
            fillcolor=f'rgba({color[0]}, {color[1]}, {color[2]}, {0.8 if is_vibration_source else 0.3})',
            line=dict(
                color=f'rgb({color[0]}, {color[1]}, {color[2]})',
                width=6 if is_vibration_source else 2
            ),
            name=f'🎯 VIBRATION SOURCE: {label}' if is_vibration_source else f'{label}',
            mode='lines',
            hovertemplate=f'<b>{"🎯 VIBRATION SOURCE" if is_vibration_source else "Shape"}</b><br>Center: ({center[0]}, {center[1]})<br>Radius: 3.0<extra></extra>'
        ))
        
        # Add pulsing border for vibration source
        if is_vibration_source:
            fig.add_trace(go.Scatter(
                x=x_circle, y=y_circle,
                line=dict(color='orange', width=3, dash='dash'),
                mode='lines',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Center marker
        fig.add_trace(go.Scatter(
            x=[center[0]], y=[center[1]],
            mode='markers+text',
            marker=dict(
                size=20 if is_vibration_source else 10,
                color=f'rgb({color[0]}, {color[1]}, {color[2]})',
                symbol='star' if is_vibration_source else 'circle',
                line=dict(color='white', width=3 if is_vibration_source else 1)
            ),
            text=['🎯' if is_vibration_source else label],
            textposition='middle center',
            textfont=dict(size=14 if is_vibration_source else 10, color='white'),
            showlegend=False
        ))
    
    fig.update_layout(
        title='🎯 2D Vibration Source Identification - Shapes Only',
        xaxis=dict(title='X Position', showgrid=True, range=[-2, 42]),
        yaxis=dict(title='Y Position', showgrid=True, scaleanchor='x', scaleratio=1, range=[-2, 25]),
        showlegend=True,
        width=800, height=600,
        plot_bgcolor='white'
    )
    
    return fig

def create_3d_clean_plot(centers, closest_idx, colors, labels, radius, height):
    """Create 3D cylinder plot with vibration source highlighting"""
    fig = go.Figure()
    
    for i, (center, color, label) in enumerate(zip(centers, colors, labels)):
        is_vibration_source = (i == closest_idx)
        
        # Generate cylinder coordinates
        theta = np.linspace(0, 2*np.pi, 50)
        z = np.linspace(-height/2, height/2, 20)
        
        theta_mesh, z_mesh = np.meshgrid(theta, z)
        x_cyl = center[0] + radius * np.cos(theta_mesh)
        y_cyl = center[1] + radius * np.sin(theta_mesh)
        z_cyl = z_mesh + center[2]
        
        # Cylinder surface
        fig.add_trace(go.Surface(
            x=x_cyl, y=y_cyl, z=z_cyl,
            colorscale=[[0, f'rgb({color[0]}, {color[1]}, {color[2]})'], 
                       [1, f'rgb({max(color[0]-50, 0)}, {max(color[1]-50, 0)}, {max(color[2]-50, 0)})']] if not is_vibration_source else
                       [[0, 'rgb(220, 0, 0)'], [0.5, 'rgb(255, 100, 0)'], [1, 'rgb(255, 0, 0)']],
            opacity=0.9 if is_vibration_source else 0.6,
            name=f'🎯 VIBRATION SOURCE: {label}' if is_vibration_source else f'{label}',
            showscale=False,
            hovertemplate=f'<b>{"🎯 VIBRATION SOURCE" if is_vibration_source else "Shape"}</b><br>Center: ({center[0]}, {center[1]}, {center[2]})<br>Radius: {radius}<br>Height: {height}<extra></extra>'
        ))
        
        # Enhanced wireframe for vibration source
        if is_vibration_source:
            fig.add_trace(go.Surface(
                x=x_cyl, y=y_cyl, z=z_cyl,
                colorscale=[[0, 'rgba(255,255,255,0.1)'], [1, 'rgba(255,255,255,0.1)']],
                opacity=0.2,
                showscale=False,
                showlegend=False,
                contours=dict(
                    x=dict(show=True, color="orange", width=3),
                    y=dict(show=True, color="orange", width=3),
                    z=dict(show=True, color="orange", width=3)
                ),
                hoverinfo='skip'
            ))
        
        # Center marker
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode='markers+text',
            marker=dict(
                size=20 if is_vibration_source else 12,
                color=f'rgb({color[0]}, {color[1]}, {color[2]})',
                symbol='diamond' if is_vibration_source else 'circle',
                line=dict(color='white', width=4 if is_vibration_source else 2)
            ),
            text=['🎯 VIBRATION<br>SOURCE' if is_vibration_source else label],
            textposition='top center',
            textfont=dict(size=12 if is_vibration_source else 10, color='white'),
            showlegend=False
        ))
    
    fig.update_layout(
        title='🗂️ 3D Vibration Source Identification - Cylinders Only',
        scene=dict(
            xaxis=dict(title='X Position', range=[-2, 42], showgrid=True),
            yaxis=dict(title='Y Position', range=[-2, 25], showgrid=True),
            zaxis=dict(title='Z Position', range=[-4, 4], showgrid=True),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.5)),
            bgcolor='rgba(240,240,240,0.5)',
            aspectmode='cube'
        ),
        showlegend=True,
        width=1000, height=700
    )
    
    return fig

# ===== STEP 2: INTEGRATE INTO YOUR EXISTING STREAMLIT APP =====

def main():
    st.title("🎯 Your Enhanced Vibration Localization App")
    
    # ... your existing code for file uploads, sensor processing, etc. ...
    
    # After you have:
    # - available_sensors (list of processed sensors)
    # - estimated_location (calculated source location)
    # - shapes (list of geometric shapes)
    
    if st.button("🚀 Perform Advanced Localization"):
        # ... your existing processing code ...
        
        # After localization calculation:
        if estimated_location is not None:
            # Find closest shape to estimated location
            closest_shape_idx, min_distance = find_closest_shape(estimated_location, shapes)
            
            # === THIS IS WHERE YOU ADD THE CLEAN VISUALIZATION ===
            st.subheader("🎯 Vibration Source Identification")
            
            # Show results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Estimated Location:** ({estimated_location[0]:.2f}, {estimated_location[1]:.2f}, {estimated_location[2]:.2f})")
            with col2:
                st.error(f"**🎯 VIBRATION SOURCE:** Shape {closest_shape_idx + 1} (Distance: {min_distance:.2f} units)")
            
            # Create and display the clean visualization
            fig = create_clean_3d_visualization(closest_shape_idx, len(available_sensors))
            st.plotly_chart(fig, use_container_width=True)
            
            # Final identification message
            st.markdown(f"""
            <div style="background: linear-gradient(45deg, #ff6b6b, #ffa500); color: white; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; margin: 20px 0;">
            🎯 VIBRATION SOURCE SUCCESSFULLY IDENTIFIED!<br>
            📍 Shape {closest_shape_idx + 1} at location ({shapes[closest_shape_idx].center[0]}, {shapes[closest_shape_idx].center[1]}, {shapes[closest_shape_idx].center[2]})<br>
            🎯 Confidence: {min_distance:.2f} units distance from estimated source
            </div>
            """, unsafe_allow_html=True)

# ===== STEP 3: HOW USERS WILL INTERACT =====

"""
👥 USER EXPERIENCE FLOW:

1. 📁 User uploads sensor files (vibration_data11.1.txt, etc.)
2. ⚙️ User sets coordinates and parameters in sidebar
3. 🚀 User clicks "Perform Advanced Localization"
4. 🔍 System processes files and detects sensor count
5. 📊 System shows:
   - 3 sensors → 2D circle plot
   - 4+ sensors → 3D cylinder plot
6. 🎯 System highlights vibration source in RED
7. 📈 User sees interactive visualization:
   - Mouse drag = rotate (3D only)
   - Mouse scroll = zoom
   - Right-click drag = pan
8. ✅ User gets final identification message

🎯 WHAT USERS SEE:
- Clean geometric shapes (no calculation points)
- One shape highlighted in bright red
- Clear "VIBRATION SOURCE" labeling
- Interactive 3D controls for exploration
- Professional results summary
"""

# ===== STEP 4: CUSTOMIZATION OPTIONS =====

def customize_visualization():
    """
    🎨 CUSTOMIZATION TIPS:
    
    1. Change shape colors:
       shape_colors = [[your_r, your_g, your_b], ...]
    
    2. Modify shape locations:
       default_centers = [[your_x, your_y, your_z], ...]
    
    3. Adjust shape sizes:
       default_radius = your_radius
       default_height = your_height
    
    4. Change vibration source highlighting:
       - Modify opacity values (0.9 for source, 0.6 for others)
       - Change colors (red gradient for source)
       - Adjust marker sizes (20 for source, 12 for others)
    
    5. Add more shapes:
       - Increase list lengths in default_centers, colors, labels
       - System will automatically handle any number of shapes
    """
    pass

if __name__ == "__main__":
    main()