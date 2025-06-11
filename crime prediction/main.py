import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import timedelta
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import LabelEncoder
import re

# Configure page
st.set_page_config(layout="wide", page_title="Indian Crime Analytics", page_icon="🚓")
st.title("🔍 Crime Analytics Dashboard (2020-2024)")

# === Load and Preprocess Data ===
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("crime_dataset_india.csv")
        
        # Debug: Show raw data sample
        st.sidebar.subheader("Data Sample")
        st.sidebar.write(df[["Date of Occurrence", "Time of Occurrence"]].head(3))
        
        # Improved datetime parsing
        def parse_datetime(row):
            # Extract date part from Date of Occurrence (DD-MM-YYYY)
            date_str = str(row["Date of Occurrence"]).strip()
            date_part = date_str.split()[0] if " " in date_str else date_str
            
            # Extract time part from Time of Occurrence
            time_str = str(row["Time of Occurrence"]).strip()
            
            # Handle different time formats
            time_match = re.search(r"(\d{1,2}:\d{2})", time_str)
            if time_match:
                time_part = time_match.group(1)
            else:
                time_part = "00:00"  # Default to midnight if no time found
            
            # Handle date parsing with multiple formats
            try:
                # Try explicit format first
                return pd.to_datetime(f"{date_part} {time_part}", format="%d-%m-%Y %H:%M")
            except:
                try:
                    # Try dayfirst parser
                    return pd.to_datetime(f"{date_part} {time_part}", dayfirst=True)
                except:
                    try:
                        # Try without time
                        return pd.to_datetime(date_part, dayfirst=True)
                    except:
                        return pd.NaT
        
        # Apply the parser
        df["Occurrence_Datetime"] = df.apply(parse_datetime, axis=1)
        
        # Feature engineering
        df["Year"] = df["Occurrence_Datetime"].dt.year
        df["Month"] = df["Occurrence_Datetime"].dt.to_period("M").dt.to_timestamp()
        df["Hour"] = df["Occurrence_Datetime"].dt.hour
        df["Day_of_Week"] = df["Occurrence_Datetime"].dt.day_name()
        df["Quarter"] = df["Occurrence_Datetime"].dt.quarter
        
        # Clean and standardize data
        text_cols = ["City", "Crime Description", "Victim Gender", "Weapon Used", "Crime Domain"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.title().str.strip()
                df[col] = df[col].replace({'Nan': 'Unknown', 'None': 'Unknown', '': 'Unknown'})
        
        # Encode categorical features
        if "Crime Description" in df.columns:
            le = LabelEncoder()
            df["Crime_Code"] = le.fit_transform(df["Crime Description"])
        
        # Date diagnostics
        invalid_dates = df["Occurrence_Datetime"].isna().sum()
        if invalid_dates > 0:
            st.sidebar.warning(f"⚠️ {invalid_dates} invalid dates found")
            
            # Show problematic rows
            st.sidebar.subheader("Problematic Date Samples")
            problems = df[df["Occurrence_Datetime"].isna()][["Date of Occurrence", "Time of Occurrence"]].head(5)
            st.sidebar.write(problems)
        
        return df.dropna(subset=["Occurrence_Datetime"])
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

df = load_data()

# === Sidebar Filters ===
st.sidebar.header("🔧 Advanced Filters")

# Crime type filter
crime_options = ["All"] + (df["Crime Description"].dropna().unique().tolist() 
                          if "Crime Description" in df.columns and not df.empty else ["All"])
selected_crime = st.sidebar.selectbox("Crime Type", crime_options)

# Location filter
city_options = ["All"] + (df["City"].dropna().unique().tolist() 
                         if "City" in df.columns and not df.empty else ["All"])
selected_location = st.sidebar.selectbox("Location", city_options)

# Time filters
if not df.empty and "Year" in df.columns:
    min_year = int(df["Year"].min())
    max_year = int(df["Year"].max())
    time_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
else:
    time_range = (2020, 2024)

# Additional filters
if "Victim Gender" in df.columns and not df.empty:
    gender_options = ["All"] + df["Victim Gender"].unique().tolist()
    selected_gender = st.sidebar.selectbox("Victim Gender", gender_options)

if "Weapon Used" in df.columns and not df.empty:
    weapon_options = ["All"] + df["Weapon Used"].unique().tolist()
    selected_weapon = st.sidebar.selectbox("Weapon Used", weapon_options)

# Forecast horizon
forecast_months = st.sidebar.slider("Forecast Horizon (months)", 3, 12, 6)

# === Apply Filters ===
filtered_df = df.copy()
if selected_crime != "All":
    filtered_df = filtered_df[filtered_df["Crime Description"] == selected_crime]
if selected_location != "All":
    filtered_df = filtered_df[filtered_df["City"] == selected_location]
if "Victim Gender" in df.columns and selected_gender != "All" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["Victim Gender"] == selected_gender]
if "Weapon Used" in df.columns and selected_weapon != "All" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["Weapon Used"] == selected_weapon]
    
if "Year" in filtered_df.columns and not filtered_df.empty:
    filtered_df = filtered_df[
        (filtered_df["Year"] >= time_range[0]) & 
        (filtered_df["Year"] <= time_range[1])
    ]

# === Dashboard Layout ===
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📍 Geospatial", "⏳ Time Analysis", "🔮 Forecasting"])

with tab1:  # Overview Tab
    st.header("Crime Data Overview")
    
    if not filtered_df.empty:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Crimes", len(filtered_df))
        col2.metric("Cities Covered", filtered_df["City"].nunique() if "City" in filtered_df.columns else "N/A")
        
        if "Victim Age" in filtered_df.columns:
            avg_age = int(filtered_df["Victim Age"].mean())
            col3.metric("Avg Victim Age", avg_age)
        else:
            col3.metric("Avg Victim Age", "N/A")
            
        if 'Case Closed' in filtered_df.columns:
            clearance_rate = filtered_df['Case Closed'].value_counts(normalize=True).get('Yes', 0)*100
            col4.metric("Clearance Rate", f"{clearance_rate:.1f}%")
        else:
            col4.metric("Clearance Rate", "N/A")
        
        # Crime Distribution Visualizations
        st.subheader("Crime Distribution Analysis")
        
        # Row 1: Crime Types and Weapons
        col1, col2 = st.columns(2)
        with col1:
            if "Crime Description" in filtered_df.columns:
                st.write("**Crime Type Distribution**")
                crime_counts = filtered_df["Crime Description"].value_counts().reset_index()
                crime_counts.columns = ['Crime Description', 'count']
                fig = px.pie(crime_counts, names='Crime Description', values='count', 
                             hover_data=['count'], hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            if "Weapon Used" in filtered_df.columns:
                st.write("**Weapon Usage**")
                weapon_counts = filtered_df["Weapon Used"].value_counts().reset_index()
                weapon_counts.columns = ['Weapon Used', 'count']
                fig = px.bar(weapon_counts, x='Weapon Used', y='count', color='Weapon Used',
                             text='count', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Row 2: Temporal Patterns
        col1, col2 = st.columns(2)
        with col1:
            if "Hour" in filtered_df.columns:
                st.write("**Hourly Crime Pattern**")
                hourly = filtered_df["Hour"].value_counts().sort_index().reset_index()
                hourly.columns = ['Hour', 'count']
                fig = px.line(hourly, x='Hour', y='count', markers=True)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if "Day_of_Week" in filtered_df.columns:
                st.write("**Day of Week Pattern**")
                dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_counts = filtered_df["Day_of_Week"].value_counts().reset_index()
                dow_counts.columns = ['Day_of_Week', 'count']
                dow_counts['Day_of_Week'] = pd.Categorical(dow_counts['Day_of_Week'], categories=dow_order, ordered=True)
                dow_counts = dow_counts.sort_values('Day_of_Week')
                fig = px.bar(dow_counts, x='Day_of_Week', y='count', text='count')
                st.plotly_chart(fig, use_container_width=True)
        
        # Victim Demographics
        st.subheader("Victim Demographics")
        col1, col2 = st.columns(2)
        with col1:
            if "Victim Age" in filtered_df.columns:
                st.write("**Age Distribution**")
                fig = px.histogram(filtered_df, x="Victim Age", nbins=20, 
                                  marginal="box", color_discrete_sequence=['indianred'])
                st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            if "Victim Gender" in filtered_df.columns:
                st.write("**Gender Distribution**")
                gender_counts = filtered_df["Victim Gender"].value_counts().reset_index()
                gender_counts.columns = ['Victim Gender', 'count']
                fig = px.pie(gender_counts, names='Victim Gender', values='count', 
                            hole=0.3, color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available after filtering")

# === Geospatial Tab ===
with tab2:  # Geospatial Tab
    st.header("Geospatial Crime Analysis")
    
    if not filtered_df.empty and "City" in filtered_df.columns:
        # City-level crime heatmap
        st.subheader("Crime Density by City")
        city_crimes = filtered_df["City"].value_counts().reset_index()
        city_crimes.columns = ['City', 'Crime Count']
        
        # Get approximate coordinates for Indian cities
        city_coords = {
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.7041, 77.1025),
            'Chennai': (13.0827, 80.2707),
            'Pune': (18.5204, 73.8567),
            'Ahmedabad': (23.0225, 72.5714),
            'Ludhiana': (30.9010, 75.8573),
            'Kolkata': (22.5726, 88.3639),
            'Bangalore': (12.9716, 77.5946),
            'Hyderabad': (17.3850, 78.4867),
            'Jaipur': (26.9124, 75.7873),
            'Surat': (21.1702, 72.8311),
            'Lucknow': (26.8467, 80.9462)
        }
        
        # Add coordinates to our data
        city_crimes['Lat'] = city_crimes['City'].apply(lambda x: city_coords.get(x, (None, None))[0])
        city_crimes['Lon'] = city_crimes['City'].apply(lambda x: city_coords.get(x, (None, None))[1])
        city_crimes = city_crimes.dropna(subset=['Lat', 'Lon'])
        
        # Calculate normalized radius
        if not city_crimes.empty:
            max_crime = city_crimes['Crime Count'].max()
            city_crimes['Normalized Radius'] = 5 + 20 * (city_crimes['Crime Count'] / max_crime)
            
            # Create map with wider view
            m = folium.Map(location=[23.5, 80], zoom_start=4.5, 
                          tiles='CartoDB dark_matter', 
                          width='100%', height='500px')
            
            # Add crime markers with optimized size
            for idx, row in city_crimes.iterrows():
                folium.CircleMarker(
                    location=[row['Lat'], row['Lon']],
                    radius=row['Normalized Radius'],
                    popup=f"{row['City']}: {row['Crime Count']} crimes",
                    color='#ff5252',
                    fill=True,
                    fill_color='#ff5252',
                    fill_opacity=0.6,
                    weight=1
                ).add_to(m)
            
            folium_static(m, width=1000, height=500)
            
            # City comparison
            st.subheader("City-wise Crime Comparison")
            fig = px.bar(city_crimes.sort_values('Crime Count', ascending=False), 
                        x='City', y='Crime Count', color='Crime Count',
                        color_continuous_scale='reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid city coordinates found")
    else:
        st.warning("No location data available")

# === Time Analysis Tab ===
with tab3:
    st.subheader("📅 Time-Based Crime Trends")
    
    if not filtered_df.empty:
        # Monthly trend
        if 'Month' in filtered_df.columns:
            monthly_trend = filtered_df.groupby('Month')['Crime_Code'].count().reset_index()
            fig = px.line(monthly_trend, x='Month', y='Crime_Code',
                          title="Monthly Crime Trend", markers=True)
            st.plotly_chart(fig, use_container_width=True)

        # Year-wise trend
        if 'Year' in filtered_df.columns:
            yearly_trend = filtered_df.groupby('Year')['Crime_Code'].count().reset_index()
            fig = px.bar(yearly_trend, x='Year', y='Crime_Code',
                         title="Year-wise Crime Trend", color='Crime_Code')
            st.plotly_chart(fig, use_container_width=True)

        # Crimes per day of the week
        if 'Day_of_Week' in filtered_df.columns:
            dow_trend = filtered_df['Day_of_Week'].value_counts().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ]).reset_index()
            dow_trend.columns = ['Day', 'Crimes']

            fig = px.bar(dow_trend, x='Day', y='Crimes',
                         title="Crime Distribution Across Days of the Week", color='Crimes')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for time analysis")

# === Forecasting Tab ===
with tab4:
    st.subheader("🔮 Crime Forecasting Using ARIMA")
    
    if not filtered_df.empty and 'Occurrence_Datetime' in filtered_df.columns:
        # Create continuous monthly time series
        ts = filtered_df.set_index('Occurrence_Datetime').resample('M').size()
        ts = ts.asfreq('M').fillna(0)
        
        # Show time series diagnostics
        st.write(f"Time Range: {ts.index.min().strftime('%Y-%m')} to {ts.index.max().strftime('%Y-%m')}")
        st.write(f"Data Points: {len(ts)} months")
        
        if len(ts) < 12:
            st.warning("Need at least 12 months of data for forecasting")
            st.write("### Raw Time Series")
            st.line_chart(ts)
        else:
            try:
                # Auto-select best ARIMA parameters
                best_aic = np.inf
                best_order = None
                
                # Try different parameter combinations
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                model = ARIMA(ts, order=(p,d,q))
                                results = model.fit()
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (p,d,q)
                            except:
                                continue
                
                if best_order:
                    st.success(f"Using ARIMA{best_order} model (AIC: {best_aic:.2f})")
                    model = ARIMA(ts, order=best_order)
                    model_fit = model.fit()
                    
                    # Forecast next months
                    forecast_result = model_fit.get_forecast(steps=forecast_months)
                    forecast = forecast_result.predicted_mean
                    conf_int = forecast_result.conf_int()
                    
                    # Generate forecast dates
                    last_date = ts.index[-1]
                    forecast_dates = pd.date_range(
                        last_date + pd.DateOffset(months=1), 
                        periods=forecast_months, 
                        freq='M'
                    )
                    
                    # Create forecast DataFrame
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecasted Crimes': forecast.values,
                        'Lower CI': conf_int.iloc[:, 0],
                        'Upper CI': conf_int.iloc[:, 1]
                    })
                    
                    # Plot results
                    fig = go.Figure()
                    
                    # Actual data
                    fig.add_trace(go.Scatter(
                        x=ts.index, 
                        y=ts.values, 
                        mode='lines+markers', 
                        name='Actual Crimes',
                        line=dict(color='#1f77b4')
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'], 
                        y=forecast_df['Forecasted Crimes'],
                        mode='lines+markers', 
                        name='Forecast',
                        line=dict(color='#ff7f0e', dash='dash')
                    ))
                    
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'], 
                        y=forecast_df['Lower CI'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'], 
                        y=forecast_df['Upper CI'],
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(255, 127, 14, 0.2)',
                        line=dict(width=0),
                        name='95% Confidence'
                    ))
                    
                    fig.update_layout(
                        title=f"Crime Rate Forecast (Next {forecast_months} Months)",
                        xaxis_title="Date",
                        yaxis_title="Crime Count",
                        hovermode="x unified",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("📌 Forecasted Values:")
                    st.dataframe(forecast_df.set_index('Date').style.format("{:.1f}"))
                    
                else:
                    st.error("Failed to find suitable ARIMA parameters")
                    
            except Exception as e:
                st.error(f"Forecasting failed: {str(e)}")
                st.error("Common causes: insufficient data, non-stationary series, or parameter issues")
    else:
        st.error("Datetime column not available in filtered data")

# Footer
st.markdown("---")
st.caption("©️ 2024 Crime Analytics Dashboard")

# Add analytics
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ Data last updated: 2024-05-30")