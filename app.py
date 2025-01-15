import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import sem, t
from scipy.integrate import simpson
import streamlit as st

# Title of the App
st.title("Charging Power Graph Generator")
st.write("Upload a CSV file with the correct template format to generate the graph.")

# Download Template CSV
with open("Import_template.csv", "rb") as file:
    st.download_button(
        label="Download Template CSV",
        data=file,
        file_name="Import_template.csv",
        mime="text/csv"
    )

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Preprocess data
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], format="%d/%m/%Y %H:%M", errors='coerce')
    data['Hour'] = data['Timestamp'].dt.hour
    data['Time'] = data['Timestamp'].dt.time
    data['Date'] = data['Timestamp'].dt.date
    data['Day_of_Week'] = data['Timestamp'].dt.day_name()
    
    # Select Filter Option
    filter_option = st.selectbox("Select Time Filter", ["All Days", "Weekdays", "Weekends", "After Business Hours"])
    
    # User Inputs for Grid Capacity and Average Battery Size
    grid_capacity = st.number_input("Grid Capacity (kW)", min_value=0, value=150)
    avg_battery_size = st.number_input("Average EV Battery Size (kWh)", min_value=1, value=55)

    # Function Definitions
    def filter_data(data, filter_option):
        if filter_option == "All Days":
            return data
        elif filter_option == "Weekdays":
            return data[data['Day_of_Week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
        elif filter_option == "Weekends":
            return data[data['Day_of_Week'].isin(['Saturday', 'Sunday'])]
        elif filter_option == "After Business Hours":
            return data[(data['Hour'] >= 18) | (data['Hour'] < 9)]
        else:
            raise ValueError("Invalid filter option")
    
    def plot_average_power_used(data, filter_option, grid_capacity, avg_battery_size):
        filtered_data = filter_data(data, filter_option)
        grouped = filtered_data.groupby('Hour')['Power_Kw']
        means = grouped.mean()
        conf_intervals = grouped.apply(lambda x: t.interval(0.9, len(x)-1, loc=x.mean(), scale=sem(x)))
        lower_bounds = conf_intervals.apply(lambda x: x[0])
        upper_bounds = conf_intervals.apply(lambda x: x[1])

        # Convert to numpy arrays for plotting
        x = means.index.values
        y_mean = means.values
        y_lower = lower_bounds.values
        y_upper = upper_bounds.values

        # Calculate the available energy
        available_energy = np.maximum(0, grid_capacity - y_upper)
        total_available_energy = simpson(available_energy, x)
        Total_EVBattery_charged = int(total_available_energy / avg_battery_size)

        # Create Plotly graph
        fig = go.Figure()

        # Add mean line
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, mode='lines+markers',
            name='Mean Power Consumption',
            line=dict(color='royalblue', width=3),
            marker=dict(size=6)
        ))

        # Add confidence interval band
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill='toself', fillcolor='rgba(135, 206, 250, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='90% Confidence Interval'
        ))

        # Add grid capacity line
        fig.add_trace(go.Scatter(
            x=x, y=[grid_capacity] * len(x),
            mode='lines', name='Grid Capacity',
            line=dict(color='darkgrey', dash='dash', width=4)
        ))

        # Add banner
        banner_text = (f"Based on your historical data you have {total_available_energy:.2f} kWh available per day <br>"
                       f"This is equivalent to charging {Total_EVBattery_charged} EVs")
        fig.add_annotation(
            text=banner_text,
            xref="paper", yref="paper",
            x=1, y=1,
            xanchor="right", yanchor="top",
            showarrow=False,
            font=dict(size=14, color="white"),
            align="center",
            bgcolor="rgba(50, 50, 50, 0.8)"
        )

        # Layout customization
        fig.update_layout(
            title=f"Power Consumption Confidence Interval ({filter_option})",
            xaxis_title="Hour of Day",
            yaxis_title="Power (kW)",
            template="plotly_dark",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            yaxis=dict(range=[0, None]),
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
        )

        st.plotly_chart(fig)

    # Button to Generate Graph
    if st.button("Generate Graph"):
        plot_average_power_used(data, filter_option, grid_capacity, avg_battery_size)
