import numpy as np
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

# Initialize session state to persist data across reruns
if "lines" not in st.session_state:
    st.session_state.lines = {}  # {line_name: {x, y, color, amplitude, frequency, phase}}

# Function to create/update the Bokeh plot
def create_plot():
    p = figure(title="Dynamic Line Plot", x_axis_label='X', y_axis_label='Y', width=600, height=400)
    for name, line in st.session_state.lines.items():
        p.line(line['x'], line['y'], line_width=2, legend_label=name, color=line['color'])
    p.legend.location = "top_left"
    st.bokeh_chart(p)

# Streamlit UI
st.title("Dynamic Line Plot with Streamlit")

# Sidebar for inputs
with st.sidebar:
    st.header("Add New Line")
    name = st.text_input("Line Name", key="name_input")
    color = st.color_picker("Line Color", "#1f77b4")
    amplitude = st.slider("Amplitude", 0.1, 5.0, 1.0, step=0.1)
    frequency = st.slider("Frequency", 0.1, 5.0, 1.0, step=0.1)
    phase = st.slider("Phase", 0.0, 2 * np.pi, 0.0, step=0.1)
    
    if st.button("Add Line"):
        if name in st.session_state.lines:
            st.error("Error: Line name already exists!")
        else:
            x = np.linspace(0, 10, 100)
            y = amplitude * np.sin(frequency * x + phase)
            st.session_state.lines[name] = {
                'x': x,
                'y': y,
                'color': color,
                'amplitude': amplitude,
                'frequency': frequency,
                'phase': phase
            }
            st.success(f"Line '{name}' added successfully!")

# Main area for plot and deletion
st.header("Plot & Line Management")
create_plot()

# Delete selected lines
if st.session_state.lines:
    selected_lines = st.multiselect(
        "Select lines to delete",
        options=list(st.session_state.lines.keys()),
        key="delete_select"
    )
    if st.button("Delete Selected Lines"):
        for name in selected_lines:
            del st.session_state.lines[name]
        st.success(f"Deleted {len(selected_lines)} lines!")
else:
    st.info("No lines to delete. Add a line first!")
