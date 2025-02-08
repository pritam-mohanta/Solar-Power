import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go

# Load Data Function
@st.cache_data
def load_data():
    df_gen1 = pd.read_csv('Plant_1_Generation_Data.csv')
    df_sen1 = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
    
    df_gen1.drop('PLANT_ID', axis=1, inplace=True)
    df_sen1.drop('PLANT_ID', axis=1, inplace=True)
    
    df_gen1['DATE_TIME'] = pd.to_datetime(df_gen1['DATE_TIME'], format='%d-%m-%Y %H:%M')
    df_sen1['DATE_TIME'] = pd.to_datetime(df_sen1['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    
    df_plant1 = pd.merge(df_gen1, df_sen1.drop(columns=['SOURCE_KEY']), on='DATE_TIME')
    
    return df_gen1, df_sen1, df_plant1

df_gen1, df_sen1, df_plant1 = load_data()


# Apply enhanced CSS
st.markdown(
    """
    <style>
        .stApp { padding-top: 1px !important; } /* Reduce top padding */
        .main-title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            background: linear-gradient(to right, #FF8C00, #FFD700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-top: -72px; /* Reduce space above the title */
            margin-bottom: 2px;
        }
        .metric-box { 
            background-color: #f7f9fc; padding: 4px; 
            border-radius: 6px; box-shadow: 1px 1px 4px rgba(0, 0, 0, 0.1);
        }
        .chart-container {
            background-color: #ffffff; padding: 6px;
            border-radius: 6px; box-shadow: 1px 1px 4px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 🌞 Main Title ---
st.markdown("<h1 class='main-title'>🌞 Solar Power Plant Analysis</h1>", unsafe_allow_html=True)

# --- 📅 Sidebar ---
st.sidebar.header("📅 Date Selection")

# Ensure DATE_TIME is in datetime format
df_plant1["DATE_TIME"] = pd.to_datetime(df_plant1["DATE_TIME"], errors="coerce")
df_plant1["DATE"] = df_plant1["DATE_TIME"].dt.date

min_date, max_date = df_plant1["DATE"].min(), df_plant1["DATE"].max()

date_selection = st.sidebar.date_input("Select Date Range:", [min_date, max_date], min_value=min_date, max_value=max_date)

# Ensure valid selection
if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2:
    start_date, end_date = date_selection
else:
    st.warning("⚠️ Select a valid date range.")
    st.stop()

# --- 🔍 Filter Data ---
df_filtered = df_plant1[(df_plant1["DATE"] >= start_date) & (df_plant1["DATE"] <= end_date)]

# 📊 KPI Calculations
total_dc_power = df_filtered["DC_POWER"].sum() / 1_000_000
total_ac_power = df_filtered["AC_POWER"].sum() / 1_000_000
avg_ambient_temp = df_filtered["AMBIENT_TEMPERATURE"].mean()
avg_module_temp = df_filtered["MODULE_TEMPERATURE"].mean()
total_irradiation = df_filtered["IRRADIATION"].sum()
unique_sources = df_filtered["SOURCE_KEY"].nunique()

# --- 🏆 KPI Display ---
col1, col2, col3 = st.columns(3)
col1.metric("🔋 DC Power", f"{total_dc_power:.2f} GW")
col2.metric("⚡ AC Power", f"{total_ac_power:.2f} GW")
col3.metric("🏭 Unique Plants", unique_sources)

col4, col5, col6 = st.columns(3)
col4.metric("🌡️ Ambient Temp", f"{avg_ambient_temp:.1f}°C")
col5.metric("📏 Module Temp", f"{avg_module_temp:.1f}°C")
col6.metric("☀️ Irradiation", f"{total_irradiation:.0f} kWh/m²")

# --- 📊 Power Distribution ---
df_power_distribution = df_filtered.groupby("DATE")[["AC_POWER", "DC_POWER"]].sum().reset_index()

col1, col2 = st.columns(2)

fig_ac = px.pie(df_power_distribution, values="AC_POWER", names="DATE",
    title="⚡ AC Power Distribution", color_discrete_sequence=px.colors.sequential.Blues, hole=0.4, height=350)
col1.plotly_chart(fig_ac, use_container_width=True)

fig_dc = px.pie(df_power_distribution, values="DC_POWER", names="DATE",
    title="🔋 DC Power Distribution", color_discrete_sequence=px.colors.sequential.Oranges, hole=0.4, height=350)
col2.plotly_chart(fig_dc, use_container_width=True)
# __________________________________________________________________________________________________________________________________

# Sidebar Options
st.sidebar.header("🔍 Data Exploration Options")
select_all = st.sidebar.checkbox("✅ Explore All")

# Correlation Heatmap
if select_all or st.sidebar.checkbox("🎨 Show Correlation Heatmap"):
    st.subheader("🔥 Correlation Heatmap")
    plt.figure(figsize=(10, 5))
    c = df_sen1.corr(numeric_only=True)
    sns.heatmap(c, cmap="coolwarm", annot=True)
    st.pyplot(plt)

# DC and AC Power Trends
if select_all or st.sidebar.checkbox("⚡ Show DC and AC Power Trends"):
    st.subheader("⚡ DC and AC Power Trends")
    df_gen1["time"] = df_gen1["DATE_TIME"].dt.strftime('%H:%M')
    
    fig1, ax1 = plt.subplots(figsize=(16, 6), dpi=120)
    ax1.plot(df_gen1["DATE_TIME"], df_gen1["DAILY_YIELD"], color="gold", marker="^", markersize=6, linestyle="-", linewidth=1.8, alpha=0.9)
    ax1.set_title("📅 Daily Yield over Time", fontsize=16, fontweight="bold", color="#FF8C00")
    ax1.set_xlabel("Time", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Daily Yield (kWh)", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(16, 6), dpi=120)
    ax2.plot(df_gen1["time"], df_gen1["DC_POWER"], color="blue", marker="^", markersize=6, linestyle="-", linewidth=1.8, alpha=0.9, label="🔋 DC Power")
    ax2.plot(df_gen1["time"], df_gen1["AC_POWER"], color="green", marker="^", markersize=6, linestyle="-", linewidth=1.8, alpha=0.9, label="⚡ AC Power")
    ax2.set_title("⚙️ DC and AC Power over Time", fontsize=16, fontweight="bold", color="#006400")
    ax2.set_xlabel("Time", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Power (kW)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=90)
    st.pyplot(fig2)

# DC Power and Daily Yield per Day
if select_all or st.sidebar.checkbox("🔋 Show DC Power and Daily Yield per Day"):
    st.subheader("🔋 DC Power & 🌞 Daily Yield Per Day")
    temp_gen1 = df_gen1.copy()
    temp_gen1["time"] = temp_gen1["DATE_TIME"].dt.strftime('%H:%M')
    temp_gen1["day"] = temp_gen1["DATE_TIME"].dt.date
    
    dc_power_data = temp_gen1.groupby(["time", "day"])["DC_POWER"].mean().unstack()
    daily_yield_data = temp_gen1.groupby(["time", "day"])["DAILY_YIELD"].mean().unstack()
    
    num_days = len(dc_power_data.columns)
    fig, axes = plt.subplots(nrows=num_days, ncols=1, figsize=(16, num_days * 4), dpi=100)
    if num_days == 1:
        axes = [axes]
    
    for i, column in enumerate(dc_power_data.columns):
        axes[i].plot(dc_power_data.index, dc_power_data[column], label="⚡ DC Power", color="blue", marker="^", markersize=6, linestyle="-", linewidth=1.5, alpha=0.8)
        axes[i].plot(daily_yield_data.index, daily_yield_data[column], label="🌞 Daily Yield", color="orange", marker="o", markersize=6, linestyle="-", linewidth=1.5, alpha=0.8)
        
        axes[i].set_title(f"📅 Day: {column}", fontsize=14, fontweight="bold")
        axes[i].set_xlabel("Time", fontsize=12)
        axes[i].set_ylabel("Power (kW)", fontsize=12)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, linestyle="--", alpha=0.5)
        tick_indices = np.linspace(0, len(dc_power_data.index) - 1, min(10, len(dc_power_data.index)), dtype=int)
        axes[i].set_xticks(dc_power_data.index[tick_indices])
        axes[i].set_xticklabels(dc_power_data.index[tick_indices], rotation=45, ha="right")
    
    plt.tight_layout()
    st.pyplot(fig)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Sidebar for KPI selection
st.sidebar.header("📊 KPI Selection")
kpi_options = [
    "🔋 Total Power Generated Per Day", "⏳ Average Hourly Power Output", "🌞 Peak Power Generation Hours",
    "⚡ Efficiency Analysis", "⚠️ Power Loss Analysis", "🔥 Energy Loss due to Temperature",
    "🌡️ Impact of Temperature on Power Generation"]
select_all = st.sidebar.checkbox("Select All KPIs")
selected_kpis = kpi_options if select_all else [opt for opt in kpi_options if st.sidebar.checkbox(opt)]

#KPI 1
if "🔋 Total Power Generated Per Day" in selected_kpis:
    st.subheader("🔋  Total AC Power Generated Per Day")
    
    # Group by date and sum the AC power
    df_daily_power = df_filtered.groupby("DATE")["AC_POWER"].sum().reset_index()
    
    # Create the bar chart
    fig = px.bar(df_daily_power, x="DATE", y="AC_POWER", color="AC_POWER")
    
    # Increase the frequency of the dates on the x-axis
    fig.update_xaxes(
        tickmode='linear',  # Set the tick mode to linear
        dtick=86400000.0,   # Set the interval between ticks (in milliseconds, 86400000 ms = 1 day)
        tickformat="%Y-%m-%d"  # Format the date display
    )
    
    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

#KPI 2
# Ensure 'HOUR' column exists by extracting it from 'DATE_TIME'
df_filtered["HOUR"] = df_filtered["DATE_TIME"].dt.hour

if "⏳ Average Hourly Power Output" in selected_kpis:
    st.subheader("⏳ Average Hourly Power Output")
    
    # Calculate average hourly AC and DC power
    df_hourly_avg = df_filtered.groupby("HOUR")[["AC_POWER", "DC_POWER"]].mean().reset_index()

    # Create a plot
    fig = go.Figure()

    # Add AC Power trace
    fig.add_trace(go.Scatter(
        x=df_hourly_avg["HOUR"], 
        y=df_hourly_avg["AC_POWER"], 
        mode='lines+markers', 
        name='AC Power'
    ))

    # Add DC Power trace
    fig.add_trace(go.Scatter(
        x=df_hourly_avg["HOUR"], 
        y=df_hourly_avg["DC_POWER"], 
        mode='lines+markers', 
        name='DC Power'
    ))

    # Add proper labels and title
    fig.update_layout(
        title="Average Hourly Power Output (AC vs DC)",
        xaxis_title="Hour of the Day",
        yaxis_title="Power (Kw)",
        legend_title="Power Type",
        template="plotly_dark",  
        showlegend=True
    )

    # Increase the frequency of the hours on the x-axis
    fig.update_xaxes(
        tickmode='linear',  
        dtick=1,            
        tick0=0,            
        range=[0, 23]       
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

# KPI-3: Peak Power Generation Hours
if "🌞 Peak Power Generation Hours" in selected_kpis:
    st.subheader("🌞 Peak Power Generation Hours")
    
    # Use the same df_hourly_avg from KPI-7 to ensure consistency
    peak_hour = df_hourly_avg.loc[df_hourly_avg["AC_POWER"].idxmax(), "HOUR"]

    st.write(f"**🌟 Peak Hour:** {peak_hour}:00")
    
    # Create line plot
    fig = px.line(df_hourly_avg, x="HOUR", y="AC_POWER", markers=True)

    # Add vertical line at peak hour
    fig.add_vline(x=peak_hour, line_dash="dash", line_color="red", annotation_text="Peak Hour", 
                  annotation_position="top")
    
    # Increase the frequency of the hours on the x-axis
    fig.update_xaxes(
        tickmode='linear',  
        dtick=1,            
        tick0=0,            
        range=[0, 23]       
    )

    st.plotly_chart(fig, use_container_width=True)

# KPI 4
if "⚡ Efficiency Analysis" in selected_kpis:
    st.subheader("⚡ Efficiency Analysis")
    
    # Calculate efficiency and handle infinities
    df_filtered["EFFICIENCY"] = (df_filtered["AC_POWER"] / df_filtered["DC_POWER"]).replace([np.inf, -np.inf], np.nan) * 100
    
    # Group by date and calculate mean efficiency
    df_efficiency_daily = df_filtered.groupby("DATE")["EFFICIENCY"].mean().reset_index()
    
    # Create the line chart
    fig = px.line(df_efficiency_daily, x="DATE", y="EFFICIENCY", title="⚡ Daily Efficiency")
    
    # Increase the frequency of the dates on the x-axis
    fig.update_xaxes(
        tickmode='linear',  # Set the tick mode to linear
        dtick=86400000.0,   # Set the interval between ticks (in milliseconds, 86400000 ms = 1 day)
        tickformat="%Y-%m-%d"  # Format the date display
    )
    
    # Add percentage sign to the y-axis values
    fig.update_yaxes(
        ticksuffix="%"  # Add a percentage sign to the y-axis values
    )
    
    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

#KPI 5   
if "⚠️ Power Loss Analysis" in selected_kpis:
    st.subheader("⚠️ Power Loss Analysis")
    
    # Calculate power loss
    df_filtered["POWER_LOSS"] = df_filtered["DC_POWER"] - df_filtered["AC_POWER"]
    
    # Group by date
    df_loss_daily = df_filtered.groupby("DATE")["POWER_LOSS"].mean().reset_index()

    # Create the bar chart
    fig = px.bar(df_loss_daily, x="DATE", y="POWER_LOSS", color="POWER_LOSS", title="⚠️ Daily Power Loss")

    # Increase date frequency on the x-axis
    fig.update_xaxes(
        tickmode="array",   # Use array mode to control tick frequency
        tickvals=df_loss_daily["DATE"],  # Set tick values for all dates
        tickangle=90,       # Rotate labels for better visibility
        showgrid=True       # Display grid for clarity
    )

    # Display the updated chart
    st.plotly_chart(fig, use_container_width=True)

    
# KPI 6
if "🔥 Energy Loss due to Temperature" in selected_kpis:
    st.subheader("🔥 Energy Loss due to Temperature")

    # Group data by date and calculate sum of AC power and mean ambient temperature
    daily_data = df_filtered.groupby('DATE').agg({'AC_POWER': 'sum', 'AMBIENT_TEMPERATURE': 'mean'}).reset_index()

    # Calculate energy loss due to temperature (ensure positive values)
    temp_loss_factor = 0.004
    daily_data['TEMP_ENERGY_LOSS'] = ((daily_data['AMBIENT_TEMPERATURE'] - daily_data['AMBIENT_TEMPERATURE'].mean()).clip(lower=0)) * temp_loss_factor * daily_data['AC_POWER'].mean()

    # Identify high energy loss periods (top 25% highest losses)
    high_loss_threshold = daily_data['TEMP_ENERGY_LOSS'].quantile(0.75)
    high_loss_periods = daily_data[daily_data['TEMP_ENERGY_LOSS'] > high_loss_threshold]

    # Display high loss periods
    st.write(f"🔥 High Energy Loss Periods: {len(high_loss_periods)}")
    st.write(high_loss_periods)

    # Plot energy loss due to temperature
    fig = px.line(daily_data, x="DATE", y="TEMP_ENERGY_LOSS", 
                  title="🔥 Energy Loss Due to Temperature", markers=True, line_shape='spline')

    fig.update_layout(
        xaxis_title="Date", 
        yaxis_title="Energy Loss (kWh)",
        xaxis=dict(tickangle=-45, tickmode="linear", dtick=86400000*2, tickformat="%Y-%m-%d"),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


# KPI-7 & KPI-8: Impact of Temperature on Power Generation
if "🌡️ Impact of Temperature on Power Generation" in selected_kpis:
    st.subheader(" Impact of Temperature on Power Generation")
    df_daytime = df_filtered[(df_filtered["HOUR"] >= 6) & (df_filtered["HOUR"] <= 18)]
    df_valid = df_daytime[(df_daytime["AC_POWER"] > 0) | (df_daytime["IRRADIATION"] > 0)]
    st.subheader("🌡️ Ambient Temperature vs Power Output")
    fig1 = px.scatter(df_valid, x="AMBIENT_TEMPERATURE", y="AC_POWER")
    st.plotly_chart(fig1)
    correlation_temp = df_filtered["AMBIENT_TEMPERATURE"].corr(df_filtered["AC_POWER"])
    st.write(f"📉 Correlation between Ambient Temperature and Power Output: {correlation_temp:.2f}")
    st.subheader("🔧 Module Temperature vs Power Output")
    fig2 = px.scatter(df_filtered, x="MODULE_TEMPERATURE", y="AC_POWER")
    st.plotly_chart(fig2)
    correlation_module_temp = df_filtered["MODULE_TEMPERATURE"].corr(df_filtered["AC_POWER"])
    st.write(f"📉 Correlation between Module Temperature and Power Output: {correlation_module_temp:.2f}")

# -------------------------
# Custom Styling
# -------------------------


st.header("📝 Feedback: Get In Touch With 🌞 Solar Power Plant Analysis!")

# Enhanced Contact Form with Stylish Design
contact_form = """
    <form action="https://formsubmit.co/abhishekfbd0210@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your Name" required>
        <input type="email" name="email" placeholder="Your Email" required>
        <textarea name="message" placeholder="Your Message Here..." required></textarea>
        <button type="submit">🚀 Send Message</button>
    </form>
"""

st.markdown(contact_form, unsafe_allow_html=True)

# Enhanced Local CSS for a Modern Look
def local_css():
    st.markdown("""
        <style>
            /* Form Container */
            form {
                background: linear-gradient(135deg, #1E1E1E, #333333);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 5px 15px rgba(255, 215, 0, 0.4); /* Gold Glow */
                text-align: center;
            }

            /* Input Fields */
            input, textarea {
                width: 100%;
                padding: 12px;
                margin: 8px 0;
                border: none;
                border-radius: 5px;
                background: rgba(255, 255, 255, 0.1);
                color: white;
                font-size: 16px;
                outline: none;
            }

            /* Placeholder Text Color */
            ::placeholder {
                color: #D3D3D3;
            }

            /* Submit Button */
            button {
                background: linear-gradient(135deg, #FFD700, #FF8C00); /* Gold to Orange */
                color: black;
                font-size: 18px;
                padding: 12px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: all 0.3s ease-in-out;
                box-shadow: 0px 4px 10px rgba(255, 215, 0, 0.5);
            }

            /* Hover Effect */
            button:hover {
                background: linear-gradient(135deg, #FF8C00, #FFD700);
                transform: scale(1.05);
                box-shadow: 0px 6px 15px rgba(255, 165, 0, 0.6);
            }
        </style>
    """, unsafe_allow_html=True)

# Apply the CSS Styling
local_css()

st.markdown("""# :male-student: About Section - 
This Streamlit dashboard analyzes🌞 Solar Power Plant Analysis with interactive visualizations. It includes filters, trends,Optimized with custom styling and KPI metrics, it ensures an engaging user experience.
Done By
\n:one: Pritam Mohanta
\n:two: Ritesh Patil
\n:three: Abhishekh Choudhari
\n:four: Gowtham Aryan
\n:five: Vishal Pawar
\n Thanks :heartpulse:
""")
