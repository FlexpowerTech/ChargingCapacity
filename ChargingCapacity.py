import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import sem, t
from scipy.integrate import simpson

# Set parameters
grid_capacity = 150
Confidence = 0.9 
Average_EVBattery = 55

# Read the CSV file
data = pd.read_csv('Import_template.csv')

data['Timestamp'] = pd.to_datetime(data['Timestamp'], format="%d/%m/%Y %H:%M")
print(data.dtypes)
print(data.head())

data['Time'] = data['Timestamp'].dt.time #Extract Time
data['Date'] = data['Timestamp'].dt.date #Extract Date
data['Day_of_Week'] = data['Timestamp'].dt.day_name() #Extract day of the week
data['Hour'] = data['Timestamp'].dt.hour #extract hour
print(data.head())



#Define filter options
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
    
# Plot confidence interval
def plot_average_power_used(data,filter_option):
    filtered_data = filter_data(data,filter_option)
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

    # Calculate the available energy (area between grid capacity and upper bound)
    available_energy = np.maximum(0, grid_capacity - y_upper)  # Only consider positive differences
    total_available_energy = simpson(available_energy, x)  # Integrate using Simpson's rule
    Total_EVBattery_charged = int(total_available_energy/Average_EVBattery)

    #    # Create Plotly graph
    fig = go.Figure()

    # Add mean line
    fig.add_trace(go.Scatter(
        x=x, y=y_mean, 
        mode='lines+markers', 
        name='Mean Power Consumption',
        line=dict(color='royalblue', width=3),
        marker=dict(size=6)
    ))

    # Add confidence interval band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]), 
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(135, 206, 250, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% Confidence Interval',
        showlegend=True
    ))

    # Add grid capacity line
    fig.add_trace(go.Scatter(
        x=x, y=[grid_capacity] * len(x),
        mode='lines',
        name='Grid Capacity',
        line=dict(color='darkgrey', dash='dash', width=4)
    ))

    # Add a banner with available energy information
    banner_text = (f"Based on your historical data you have {total_available_energy:.2f} kWh available per day <br>"
                    f"This is equavalent to charging {Total_EVBattery_charged} EVs")
    fig.add_annotation(
        text=banner_text,
        xref="paper", yref="paper",
        # x=1, y=1,  # Position: top-right corner
        xanchor="center", yanchor="top",
        showarrow=True,
        font=dict(size=34, color="white"),
        align="center",
        bordercolor="grey",
        borderwidth=2,
        borderpad=10,
        bgcolor="rgba(220,220,220)",  # Shadowed background
        opacity = 0.85
    )
    x_ticks = list(range(25))  # 0 to 24 for every hour
    x_labels = [f"{hour:02}:00" if hour % 2 != 0 else "" for hour in x_ticks]

    fig.update_layout(
        # Title styling
        template='plotly_dark',
        title=dict(
            text=f"<b>Power Consumption Confidence Interval ({filter_option})</b>",
            font=dict(size=38, color="white", family="Arial")  # Increase font size
        ),
        # Axis labels styling
        xaxis_title=dict(
            text="<b>Hour of Day</b>",
            font=dict(size=26, color="white")  # Increase font size
        ),
        yaxis_title=dict(
            text="<b>Power (kW)</b>",
            font=dict(size=26, color="white")  # Increase font size
        ),
        # Axis ticks styling
        xaxis=dict(
            tickfont=dict(size=24),  # X-axis ticks font size
            range=[0, 23],            # Fix the range
            tickmode='array',         # Use specific tick values and labels
            tickvals=x_ticks,         # Set a tick for every hour
            ticktext=x_labels      # Use the custom labels (only odd hours)
        ),
        yaxis=dict(
            tickfont=dict(size=24),  # Y-axis ticks font size
            range=[0, None]  # Start at 0
        ),
        # Legend styling
        legend=dict(
            x=0.5,
            y=0.2,
            xanchor="center",
            yanchor="top",
            orientation="h",
            font=dict(size=24),  # Increase legend font size
            bgcolor="rgba(50,50,50,0.8)",
            bordercolor="white",
            borderwidth=1
        ),
        margin=dict(
            b=100  # Add extra bottom margin to make room for the legend
    )
    )

    # Show the interactive plot
    fig.show()

# Example: Visualize for Weekdays
plot_average_power_used(data, filter_option="All Days")


