import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.lines import Line2D

# Load datasets
day_data = pd.read_csv("data/day.csv")
hour_data = pd.read_csv("data/hour.csv")

# Convert dates
day_data['dteday'] = pd.to_datetime(day_data['dteday'])
hour_data['dteday'] = pd.to_datetime(hour_data['dteday'])

min_date = day_data['dteday'].min()
max_date = day_data['dteday'].max()

with st.sidebar:
    # Add logo
    st.image("https://static.thenounproject.com/png/7093429-84.png")
    
    try:
        # Input date range for filtering data
        start_date, end_date = st.date_input(
            label='Time Span',
            min_value=min_date,
            max_value=max_date,
            value=[min_date, max_date]
        )
    except ValueError:
        st.text("Please select a valid date range.")

# Filter the data
try:
    filtered_day_df = day_data[
        (day_data['dteday'] >= pd.Timestamp(start_date)) & 
        (day_data['dteday'] <= pd.Timestamp(end_date))]

    filtered_hour_df = hour_data[
        (hour_data['dteday'] >= pd.Timestamp(start_date)) & 
        (hour_data['dteday'] <= pd.Timestamp(end_date))]
except NameError:
    st.error("Invalid date range. Please select a valid date range.")

def significant_drop(day_df):
    # Calculate daily changes in rental counts
    day_df['daily_change'] = day_df['cnt'].diff()

    # Identify the last significant drop (threshold: top 5% largest negative changes)
    threshold = day_df['daily_change'].quantile(0.05)  # 5th percentile (largest negative changes)
    last_significant_drop = day_df[day_df['daily_change'] < threshold].iloc[-1]  # Last significant drop

    return last_significant_drop

def plot_last_significant_drop(last_significant_drop, day_df):
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(day_df['dteday']), day_df['cnt'], label='Total Rentals', color='#009b0c', alpha=0.7)
    plt.scatter(pd.to_datetime(last_significant_drop['dteday']), last_significant_drop['cnt'], 
                color='#C70039', label=f"Last Significant Drop ({last_significant_drop['dteday']})", zorder=5)
    plt.axvline(pd.to_datetime(last_significant_drop['dteday']), color='#C70039', linestyle='--', alpha=0.7)
    plt.title('Last Significant Drop in Rental Counts', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Rental Count', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotating the last significant drop with background color and shadow
    plt.text(pd.to_datetime(last_significant_drop['dteday']), last_significant_drop['cnt'] + (last_significant_drop['cnt'] * 0.3), 
            f"{last_significant_drop['cnt']} rentals", color='#C70039', ha='center', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='#C70039', boxstyle='round,pad=0.5') # Background color with border
    )
    plt.tight_layout()

    return plt

# ---------------------------------------------------
def hourly_rentals(hour_df):
    hourly_rentals_df = hour_df.groupby('hr')['cnt'].mean()
    am_hour = hourly_rentals_df[0:12]
    pm_hour = hourly_rentals_df[12:24]

    max_am = am_hour.max() # Highest rental in the AM
    max_pm = pm_hour.max() # Highest rental in the PM

    colors_ = []
    for hour in am_hour.values:
        if hour == max_am:
            colors_.append('#388ee3')
        else:
            colors_.append('#9edaf0')

    for hour in pm_hour.values:
        if hour == max_pm:
            colors_.append('#c42200')
        else:
            colors_.append('#f0ac9e')
    
    return hourly_rentals_df, colors_

def plot_hourly_rentals(hourly_rentals_df, colors_):
    # Color for legend
    am_patch = mpatches.Patch(color='#388ee3', label='0 - 11 (Highest)')
    am_low_patch = mpatches.Patch(color='#9edaf0', label='0 - 11')
    pm_patch = mpatches.Patch(color='#c42200', label='12 - 23 (Highest)')
    pm_low_patch = mpatches.Patch(color='#f0ac9e', label='12 - 23')

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=hourly_rentals_df.index, 
        y=hourly_rentals_df.values,
        palette=colors_,
        hue=hourly_rentals_df.index)
    
    ax.set_title("Average Hourly Rentals", fontsize=16, fontweight='bold')
    ax.set_xlabel("Hour of Day", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Rentals", fontsize=12, fontweight='bold')
    ax.grid(axis='y')
    ax.legend(handles=[am_patch, am_low_patch, pm_patch, pm_low_patch], 
            title='Hour of the Day')    
    
    return fig

# --------------------------------------------------
def hourly_rentals_by_day_type(hour_df):    
    hourly_rentals_by_day_type_df = hour_df.groupby(['hr', 'workingday'])['cnt'].mean().reset_index()

    return hourly_rentals_by_day_type_df

def plot_hourly_rentals_by_day_type(hourly_rentals_by_day_type_df):
    # Separate data by working day type
    working_day_data = hourly_rentals_by_day_type_df[hourly_rentals_by_day_type_df['workingday'] == 1]
    non_working_day_data = hourly_rentals_by_day_type_df[hourly_rentals_by_day_type_df['workingday'] == 0]

    plt.figure(figsize=(12, 6))

    # Plot the lines manually for both working and non-working days
    plt.plot(working_day_data['hr'], working_day_data['cnt'], color="#28b463", lw=1, label="Working Day", marker='x', markersize=6)
    plt.plot(non_working_day_data['hr'], non_working_day_data['cnt'], color="#f69915", lw=1, label="Non-Working Day", marker='o', markersize=5)

    # Create a custom legend manually
    legend_elements = [
        Line2D([0], [0], color="#28b463", lw=2, label="Working Day", marker='x', markersize=6),
        Line2D([0], [0], color="#f69915", lw=2, label="Non-Working Day", marker='o', markersize=5),
    ]

    # Add legend
    plt.legend(handles=legend_elements, title="Working Day", loc="upper left", fontsize=10)

    # Title and labels
    plt.title('Average Bike Rentals by Hour (Working Day vs Non-Working Day)', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of the Day', fontsize=12, fontweight='bold')
    plt.ylabel('Average Rentals', fontsize=12, fontweight='bold')

    # Set x-ticks (hours of the day) and rotate labels if necessary
    plt.xticks(range(0, 24), rotation=0)

    # Add grid lines for the y-axis
    plt.grid(axis='y')

    return plt

# --------------------------------------------------
def recency(day_df):
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    recent_data = day_df.sort_values(by='dteday').tail(30)

    # Calculate daily changes
    recent_data['daily_change'] = recent_data['cnt'].diff()

    return recent_data

def plot_recency(recent_data):
    plt.figure(figsize=(12, 6))

    # Line plot for rental counts
    line_plot, = plt.plot(recent_data['dteday'], recent_data['cnt'], marker='o', label='Daily Rental Counts', color='#1251b7', alpha=0.8)

    # Highlight increasing and decreasing trends
    for i in range(1, len(recent_data)):
        if recent_data['daily_change'].iloc[i] > 0:
            plt.scatter(recent_data['dteday'].iloc[i], recent_data['cnt'].iloc[i], color='#25c651', zorder=5)
            if abs(recent_data['daily_change'].iloc[i]) > 500:  # Menyesuaikan ambang perubahan signifikan
                plt.text(recent_data['dteday'].iloc[i], recent_data['cnt'].iloc[i] + (recent_data['cnt'].iloc[i] * 0.04), 
                        f"{int(recent_data['daily_change'].iloc[i])}", color='black', ha='center', fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='#25c651', boxstyle='round,pad=0.5'))  # Background color with border
                
        elif recent_data['daily_change'].iloc[i] < 0:
            plt.scatter(recent_data['dteday'].iloc[i], recent_data['cnt'].iloc[i], color='#ce1010', zorder=5)
            if abs(recent_data['daily_change'].iloc[i]) > 500:  # Menyesuaikan ambang perubahan signifikan
                plt.text(recent_data['dteday'].iloc[i], recent_data['cnt'].iloc[i] + (recent_data['cnt'].iloc[i] * 0.09), 
                        f"{int(recent_data['daily_change'].iloc[i])}", color='black', ha='center', fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='#ce1010', boxstyle='round,pad=0.5'))  # Background color with border

    # Create dummy scatter plots for the legend
    scatter_increase = plt.scatter([], [], color='#25c651', label='Increase')
    scatter_decrease = plt.scatter([], [], color='#ce1010', label='Decrease')

    # Use elements directly from the plot for the legend
    plt.legend(handles=[line_plot, scatter_increase, scatter_decrease], labels=['Daily Rental Counts', 'Increase', 'Decrease'], fontsize=10)

    # Add a note at the bottom-left corner
    plt.text(0.05, 0, # Position relative to the figure
            "Note: Only points with daily changes > ±500 are highlighted.",
            fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'),
            transform=plt.gcf().transFigure)

    # Customizing the plot
    plt.title('Recent Trends in Bike Rentals (Last 30 Days)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Rental Count', fontsize=12, fontweight='bold')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    return plt

# --------------------------------------------------
def frequency(day_df):
    season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    day_df['season_label'] = day_df['season'].map(season_map)

    # Aggregate data to calculate average rentals by season
    seasonal_frequency = (day_df
                        .groupby('season_label')['cnt']
                        .mean()
                        .reset_index()
                        .sort_values('cnt', ascending=False))
    
    # Add a 'day' label for better visualization
    day_map = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 0: 'Sunday'}
    day_df['day_label'] = day_df['weekday'].map(day_map)

    # Define the custom order of days of the week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Convert 'day_label' column to a categorical type with specified order
    day_df['day_label'] = pd.Categorical(day_df['day_label'], categories=day_order, ordered=True)

    # Aggregate data to calculate average rentals by weekday
    weekday_frequency = (day_df
                        .groupby('day_label')['cnt']
                        .mean()
                        .reset_index()
                        .sort_values('day_label'))
    return seasonal_frequency, weekday_frequency

def plot_frequency(seasonal_frequency, weekday_frequency):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

    season_max = seasonal_frequency['cnt'].max()
    # Define colors for each season
    colors = []
    for season in seasonal_frequency['cnt'].values:
        if season == season_max:
            colors.append('#c52c2c')
        else:
            colors.append('#ffb5b5')

    sns.barplot(x='season_label', y='cnt', data=seasonal_frequency, palette=colors, hue='season_label', ax=ax[0])
    ax[0].set_title('Average Rental Count by Season', fontsize=16, fontweight='bold')
    ax[0].set_xlabel('Season', fontsize=14, fontweight='bold')
    ax[0].set_ylabel('Average Rental Count', fontsize=14, fontweight='bold')

    colors__ = ["#f55c7a", "#f56c77", "#f57c73", "#f68c70", "#f69c6d", "#f6ac69", "#f6bc65"]

    # Plot Frequency - Weekday
    sns.barplot(x='day_label', y='cnt', data=weekday_frequency, palette=colors__, hue='day_label', ax=ax[1])
    ax[1].set_title('Average Rental Count by Weekday', fontsize=16, fontweight='bold')
    ax[1].set_xlabel('Weekday', fontsize=14, fontweight='bold')
    ax[1].set_ylabel('Average Rental Count', fontsize=14, fontweight='bold')

    return fig

# --------------------------------------------------
def monetary(day_df):
    weather_sensitivity = day_df.groupby('weathersit')[['casual', 'registered']].mean().reset_index()
    length = len(weather_sensitivity)
    casual_count = weather_sensitivity['casual']
    registered_count = weather_sensitivity['registered']

    # Plot Monetary - Weather Sensitivity
    weather_labels = {1: 'Clear', 2: 'Mist', 3: 'Light Snow/Rain', 4: 'Heavy Rain/Snow'}
    weather_sensitivity['weather_label'] = weather_sensitivity['weathersit'].map(weather_labels)
    weather_name = weather_sensitivity['weather_label']

    # Calculate percentage change casual
    percent_change_casual = [0]  # First season has no previous data
    for i in range(1, length):
        change = ((casual_count[i] - casual_count[i - 1]) / casual_count[i - 1]) * 100
        percent_change_casual.append(change)

    # Calculate percentage change registered
    percent_change_registered = [0]  # First season has no previous data
    for i in range(1, length):
        change = ((registered_count[i] - registered_count[i - 1]) / registered_count[i - 1]) * 100
        percent_change_registered.append(change)

    return weather_sensitivity, weather_name, casual_count, registered_count, percent_change_casual, percent_change_registered

def plot_monetary(weather_sensitivity, weather_name, casual_count, registered_count, percent_change_casual, percent_change_registered):

    length = len(weather_sensitivity)
    # Plotting combined bar chart and line plots for casual and registered users
    plt.figure(figsize=(12, 8))

    # Barplot for 'Registered' and 'Casual'
    sns.barplot(x=weather_name, y=registered_count, data=weather_sensitivity, color='#5dade2', label='Registered Users')
    sns.barplot(x=weather_name, y=casual_count, data=weather_sensitivity, color='#2980b9', label='Casual Users')

    # Adding percentage change lines
    line_registered, = plt.plot(weather_name, registered_count, marker='o', color='#a04000', linestyle='--')
    line_casual, = plt.plot(weather_name, casual_count, marker='o', color='#a04000', linestyle='--')

    # Customizing the plot
    plt.title('Sensitivity to Weather Conditions (Casual vs Registered)', fontsize=16, weight='bold')
    plt.xlabel('Weather Condition', fontsize=12)
    plt.ylabel('Counts / Percentage Change', fontsize=12)
    plt.legend(title='Legend', fontsize=10, )
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotating the bar and line plots
    for i in range(length):
        # Annotating percentage changes
        plt.text(i, registered_count[i] + (registered_count[i] * 0.03), f"{percent_change_registered[i]:.1f}%", ha='center', color='#a04000', fontsize=12)
        plt.text(i, casual_count[i] + (casual_count[i] * 0.15), f"{percent_change_casual[i]:.1f}%", ha='center', color='#a04000', fontsize=12)

    # Creating a custom legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Add the custom label for the lines to the legend
    handles.append(line_registered)
    handles.append(line_casual)
    labels.append('Registered & Casual % Change')
    # Now we place the legend outside of the plot
    plt.legend(handles=handles, labels=labels, title='Legend', fontsize=10, loc='upper right')
    plt.tight_layout()

    return plt


# Dashboard Header
st.title("Bike Sharing Dashboard :sparkles:")
st.markdown("Analyze bike rentals with respect to time, weather, and user type.")

# Create tabs for the dashboard
tab1, tab2, tab3, tab4 = st.tabs(["Last Significant Drop", "Hourly Rentals", "Rental by Hour: Workingday vs Holiday", "RFM Analysis"])
 
with tab1:
    # --------------------------------------------
    # Section 1: Last Significant Drop
    # --------------------------------------------
    st.header("Last Significant Drop")
    st.markdown("<h6>Analysis of the most recent major drop in rentals.</h6>", unsafe_allow_html=True)

    try:
        # Total rentals and average rentals
        total_rentals = filtered_day_df['cnt'].sum()
        average_rentals = filtered_day_df['cnt'].mean()

        last_significant_drop = significant_drop(filtered_day_df)
        
        last_significant_drop_date_str = last_significant_drop['dteday'].strftime('%Y-%m-%d')

        # Create 4 columns for displaying the metrics
        col1, col2 = st.columns(2)
        # Define a custom font size for the values
        font_size = "22px"  # Adjust the size as needed
        # Displaying metrics with custom font size for values
        col1.markdown(f"<h4>Total Rentals:</h4><p style='font-size: {font_size};'>{total_rentals:,}</p>", 
                    unsafe_allow_html=True)
        col2.markdown(f"<h4>Average Rentals per Day:</h4><p style='font-size: {font_size};'>{average_rentals:,.2f}</p>", 
                    unsafe_allow_html=True)
        col1.markdown(f"<h4>Last Significant Drop Date:</h4><p style='font-size: {font_size};'>{last_significant_drop_date_str}</p>", 
                    unsafe_allow_html=True)
        col2.markdown(f"<h4>Last Significant Drop Count:</h4><p style='font-size: {font_size};'>{last_significant_drop['cnt']}</p>", 
                    unsafe_allow_html=True)

        plt = plot_last_significant_drop(last_significant_drop, filtered_day_df)
        st.pyplot(plt)
    except NameError:
        st.error("An error occurred while generating the plot.")
    except IndexError:
        st.error("No significant drop in rental counts found.")

with tab2:
    # --------------------------------------------
    # Section 2: Hourly Rentals
    # --------------------------------------------
    st.header("Hourly Rentals")
    st.markdown("<h6>Bike rental trends by hour for the selected date range.</h6>", unsafe_allow_html=True)

    try:
        hourly_rentals_df, colors_ = hourly_rentals(filtered_hour_df)
        fig = plot_hourly_rentals(hourly_rentals_df, colors_)
        st.pyplot(fig)
    except NameError:
        st.error("An error occurred while generating the plot.")

with tab3:
    # --------------------------------------------
    # Section 3: Hourly Rentals: Workingday vs Holiday
    # --------------------------------------------
    st.header("Rental by Hour: Workingday vs Holiday")
    st.markdown("<h6>Bike rental trends by hour: Workingday vs Holiday for the selected date range.</h6>", unsafe_allow_html=True)

    try:
        hourly_rentals_by_day_type_df = hourly_rentals_by_day_type(filtered_hour_df)
        plt = plot_hourly_rentals_by_day_type(hourly_rentals_by_day_type_df)
        st.pyplot(plt)
    except:
        st.error("An error occurred while generating the plot.")
        st.error("Please select dates that include both workingdays and holidays.")

with tab4:
    # --------------------------------------------
    # Section 4: RFM Analysis
    # --------------------------------------------
    st.header("RFM Analysis")

    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Recency
    # ++++++++++++++++++++++++++++++++++++++++++++++++    
    st.subheader("Recency")
    st.markdown("Recent Trends (Last 30 Days)\nHighlights significant rental changes.")

    try:
        recent_data = recency(filtered_day_df)
        plt = plot_recency(recent_data)
        st.pyplot(plt)
    except:
        st.error("An error occurred while generating the plot.")

    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Frequency
    # ++++++++++++++++++++++++++++++++++++++++++++++++
    st.subheader("Frequency")
    st.markdown("Seasonal and weekly pattern in the frequency of usage of the bikes.")
    try:
        frequency_data = frequency(filtered_day_df)
        fig = plot_frequency(*frequency_data)
        st.pyplot(fig)
    except NameError:
        st.error("An error occurred while generating the plot.")

    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Monetary
    # ++++++++++++++++++++++++++++++++++++++++++++++++
    st.subheader("Monetary")
    st.markdown("Sensitivity of casual vs registered users to weather conditions")
    try:
        monetary_data = monetary(filtered_day_df)
        plt = plot_monetary(*monetary_data)
        st.pyplot(plt)
    except NameError:
        st.error("An error occurred while generating the plot.")
