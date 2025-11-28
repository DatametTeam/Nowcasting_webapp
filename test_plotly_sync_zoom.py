"""
Test script for Plotly synchronized zoom across multiple images.
Run with: streamlit run test_plotly_sync_zoom.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("🔍 Plotly Synchronized Zoom Test")

# Generate sample data (simulating radar precipitation data)
np.random.seed(42)
height, width = 1400, 1200

# Create 3 sample "models" with similar but different patterns
def generate_sample_data(seed):
    np.random.seed(seed)
    data = np.zeros((height, width))
    # Add some "precipitation" blobs
    for _ in range(10):
        x = np.random.randint(200, width-200)
        y = np.random.randint(200, height-200)
        radius = np.random.randint(50, 150)
        intensity = np.random.uniform(5, 50)

        Y, X = np.ogrid[:height, :width]
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        mask = dist < radius
        data[mask] += intensity * (1 - dist[mask] / radius)

    return np.clip(data, 0, 200)

# Generate sample data for GT and 2 models
gt_data = generate_sample_data(42)
model1_data = generate_sample_data(43)
model2_data = generate_sample_data(44)

st.markdown("### Try zooming or panning on any plot - all plots in the row will sync! 🎯")
st.markdown("Use the tools in the top-right: Box Zoom, Pan, Reset")
st.markdown("---")

# Create synchronized plots using subplots
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Ground Truth", "Model 1", "Model 2"),
    horizontal_spacing=0.02,
    shared_xaxes=True,  # Link x-axes
    shared_yaxes=True   # Link y-axes
)

# Custom colorscale (similar to radar precipitation)
colorscale = [
    [0.0, 'rgb(255,255,255)'],   # White for no precipitation
    [0.1, 'rgb(200,200,255)'],   # Light blue
    [0.3, 'rgb(100,150,255)'],   # Medium blue
    [0.5, 'rgb(50,200,50)'],     # Green
    [0.7, 'rgb(255,255,50)'],    # Yellow
    [0.85, 'rgb(255,150,50)'],   # Orange
    [1.0, 'rgb(255,50,50)']      # Red
]

# Add heatmaps for each subplot
fig.add_trace(
    go.Heatmap(
        z=gt_data,
        colorscale=colorscale,
        zmin=0,
        zmax=200,
        showscale=True,
        colorbar=dict(title="Rain (mm/h)", x=0.32)
    ),
    row=1, col=1
)

fig.add_trace(
    go.Heatmap(
        z=model1_data,
        colorscale=colorscale,
        zmin=0,
        zmax=200,
        showscale=True,
        colorbar=dict(title="Rain (mm/h)", x=0.65)
    ),
    row=1, col=2
)

fig.add_trace(
    go.Heatmap(
        z=model2_data,
        colorscale=colorscale,
        zmin=0,
        zmax=200,
        showscale=True,
        colorbar=dict(title="Rain (mm/h)", x=0.98)
    ),
    row=1, col=3
)

# Update layout for better visualization
fig.update_layout(
    height=600,
    showlegend=False,
    title_text="Synchronized Zoom Test - Zoom on one, all follow!",
    title_x=0.5,
    hovermode='closest'
)

# Update axes to maintain aspect ratio and enable zoom
for i in range(1, 4):
    fig.update_xaxes(
        scaleanchor=f"y{i}",  # Link x to y for aspect ratio
        scaleratio=1,
        row=1, col=i
    )
    fig.update_yaxes(
        autorange='reversed',  # Flip y-axis to match image coordinates
        row=1, col=i
    )

# Display the figure
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### How it works:")
st.markdown("""
1. **`shared_xaxes=True, shared_yaxes=True`** in `make_subplots()` links all axes
2. When you zoom/pan on one subplot, Plotly automatically applies the same transformation to all linked subplots
3. **`scaleanchor`** maintains aspect ratio (like radar data)
4. **`autorange='reversed'`** flips y-axis to match image coordinates (0,0 at top-left)

**Next steps for integration:**
- Load real data from numpy arrays (we already have `gt_frame`, `pred_frame`)
- Apply the correct colormap from `configure_colorbar()`
- Add Italy shapefile overlay using `fig.add_shape()` or `fig.add_trace(go.Scattergeo())`
- Replace `st.image()` with `st.plotly_chart()` in model_comparison.py
""")

st.markdown("---")
st.markdown("### 📊 Individual Controls Test")
st.markdown("Below: Testing if we can make plots individually without subplots (one per column)")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**GT (Individual)**")
    fig1 = go.Figure(go.Heatmap(z=gt_data, colorscale=colorscale, zmin=0, zmax=200))
    fig1.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    fig1.update_xaxes(scaleanchor="y", scaleratio=1)
    fig1.update_yaxes(autorange='reversed')
    # Key: use uirevision to sync zoom state
    fig1.update_layout(uirevision='constant')
    st.plotly_chart(fig1, use_container_width=True, key="gt_individual")

with col2:
    st.markdown("**Model 1 (Individual)**")
    fig2 = go.Figure(go.Heatmap(z=model1_data, colorscale=colorscale, zmin=0, zmax=200))
    fig2.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    fig2.update_xaxes(scaleanchor="y", scaleratio=1)
    fig2.update_yaxes(autorange='reversed')
    fig2.update_layout(uirevision='constant')
    st.plotly_chart(fig2, use_container_width=True, key="model1_individual")

with col3:
    st.markdown("**Model 2 (Individual)**")
    fig3 = go.Figure(go.Heatmap(z=model2_data, colorscale=colorscale, zmin=0, zmax=200))
    fig3.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    fig3.update_xaxes(scaleanchor="y", scaleratio=1)
    fig3.update_yaxes(autorange='reversed')
    fig3.update_layout(uirevision='constant')
    st.plotly_chart(fig3, use_container_width=True, key="model2_individual")

st.markdown("**Note:** Individual plots (bottom row) do NOT sync automatically. Only the subplot version (top) syncs. We need subplots for synchronized zoom!")