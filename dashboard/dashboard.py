import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
sns.set(style='darkgrid')

# Load datasets
day_df = pd.read_csv("data/day.csv")
hour_df = pd.read_csv("data/hour.csv")

# Convert dates
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

min_date = day_df['dteday'].min()
max_date = day_df['dteday'].max()

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
        st.error("Invalid date range. Please select a valid date range.")

# Filter the data
try:
    filtered_day_df = day_df[
        (day_df['dteday'] >= pd.Timestamp(start_date)) & 
        (day_df['dteday'] <= pd.Timestamp(end_date))]

    filtered_hour_df = hour_df[
        (hour_df['dteday'] >= pd.Timestamp(start_date)) & 
        (hour_df['dteday'] <= pd.Timestamp(end_date))]
except NameError:
    st.error("Invalid date range. Please select a valid date range.")

# Dashboard Header
st.title("Bike Sharing Dashboard :sparkles:")
st.markdown("Analyze bike rentals with respect to time, weather, and user type.")


# Create tabs for the dashboard
tab1, tab2, tab3, tab4 = st.tabs(["Last Significant Drop", "Hourly Rentals", "Rental by Hour: Workingday vs Holiday", "RFM Analysis"])
 
with tab1:
    # --------------------------------------------
    # Section 1
    # --------------------------------------------
    # Section 1: Last Significant Drop
    st.header("Last Significant Drop")
    st.markdown("<h6>Analysis of the most recent major drop in rentals.</h6>", unsafe_allow_html=True)

    try:
        # Total rentals and average rentals
        total_rentals = filtered_day_df['cnt'].sum()
        average_rentals = filtered_day_df['cnt'].mean()

        # Calculate daily changes in rental counts
        filtered_day_df['daily_change'] = filtered_day_df['cnt'].diff()

        # Identify the last significant drop (threshold: top 5% largest negative changes)
        threshold = filtered_day_df['daily_change'].quantile(0.05)  # 5th percentile (largest negative changes)
        last_significant_drop = filtered_day_df[filtered_day_df['daily_change'] < threshold].iloc[-1]  # Last significant drop
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

        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(filtered_day_df['dteday']), filtered_day_df['cnt'], label='Total Rentals', color='green', alpha=0.7)
        plt.scatter(pd.to_datetime(last_significant_drop['dteday']), last_significant_drop['cnt'], 
                    color='red', label=f"Last Significant Drop ({last_significant_drop['dteday']})", zorder=5)
        plt.axvline(pd.to_datetime(last_significant_drop['dteday']), color='red', linestyle='--', alpha=0.7)
        plt.title('Last Significant Drop in Rental Counts', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Rental Count', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Annotating the last significant drop with background color and shadow
        plt.text(pd.to_datetime(last_significant_drop['dteday']), last_significant_drop['cnt'], 
                f"{last_significant_drop['cnt']} rentals", color='red', ha='center', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5') # Background color with border
        )

        plt.tight_layout()
        st.pyplot(plt)
    except NameError:
        st.error("An error occurred while generating the plot.")
    except IndexError:
        st.error("No significant drop in rental counts found.")

with tab2:
    # --------------------------------------------
    # Section 2
    # --------------------------------------------
    # Section 2: Hourly Rentals
    st.header("Hourly Rentals")
    st.markdown("<h6>Bike rental trends by hour for the selected date range.</h6>", unsafe_allow_html=True)

    try:
        # Calculate hourly rentals
        hourly_rentals = filtered_hour_df.groupby('hr')['cnt'].mean()
        am_hour = hourly_rentals[0:12]
        pm_hour = hourly_rentals[12:24]

        max_am = am_hour.max() # Highest rental in the AM
        max_pm = pm_hour.max() # Highest rental in the PM

        colors_ = []
        for hour in am_hour.values:
            if hour == max_am:
                colors_.append('#007ffc')
            else:
                colors_.append('#9edaf0')

        for hour in pm_hour.values:
            if hour == max_pm:
                colors_.append('#fb2c00')
            else:
                colors_.append('#f0ac9e')

        # Color for legend
        am_patch = mpatches.Patch(color='#007ffc', label='0 - 11 (Highest)')
        am_low_patch = mpatches.Patch(color='#9edaf0', label='0 - 11')
        pm_patch = mpatches.Patch(color='#fb2c00', label='12 - 23 (Highest)')
        pm_low_patch = mpatches.Patch(color='#f0ac9e', label='12 - 23')

        # Create a figure and axis for the hourly rentals plot
        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            x=hourly_rentals.index, 
            y=hourly_rentals.values,
            palette=colors_,
            hue=hourly_rentals.index,
            ax=ax)

        ax.set_title("Average Hourly Rentals", fontsize=16, fontweight='bold')
        ax.set_xlabel("Hour of Day", fontsize=12, fontweight='bold')
        ax.set_ylabel("Average Rentals", fontsize=12, fontweight='bold')
        ax.grid(axis='y')
        ax.legend(handles=[am_patch, am_low_patch, pm_patch, pm_low_patch], 
                title='Hour of the Day')

        st.pyplot(fig)
    except NameError:
        st.error("An error occurred while generating the plot.")

with tab3:
    # --------------------------------------------
    # Section 3
    # --------------------------------------------
    # Section 3: Hourly Rentals: Workingday vs Holiday
    st.header("Rental by Hour: Workingday vs Holiday")
    st.markdown("<h6>Bike rental trends by hour: Workingday vs Holiday for the selected date range.</h6>", unsafe_allow_html=True)

    try:
        # Calculate hourly rentals by day type
        hourly_rentals_by_day_type = filtered_hour_df.groupby(['hr', 'workingday'])['cnt'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(
            data=hourly_rentals_by_day_type,
            x='hr', y='cnt', hue='workingday', 
            style='workingday', markers=True
        )

        plt.title('Average Bike Rentals by Hour (Working Day vs Non-Working Day)', fontsize=16, fontweight='bold')
        plt.xlabel('Hour of the Day', fontsize=12, fontweight='bold')
        plt.ylabel('Average Rentals', fontsize=12, fontweight='bold')
        plt.xticks(range(0, 24), rotation=0)

        # Move legend to upper left
        sns.move_legend(ax, "upper left", title="Working Day", labels=["Non-Working Day", "Working Day"])

        plt.grid(axis='y')
        st.pyplot(plt)
    except:
        st.error("An error occurred while generating the plot.")
        st.error("Please select dates that include both workingdays and holidays.")

with tab4:
    # --------------------------------------------
    # Section 4
    # --------------------------------------------
    # Section 4: RFM Analysis
    st.header("RFM Analysis")

    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Recency
    # ++++++++++++++++++++++++++++++++++++++++++++++++    
    st.subheader("Recency")
    st.markdown("Recent Trends (Last 30 Days)\nHighlights significant rental changes.")

    try:
        # Focus on the last 30 days to observe recent trends
        recent_data = filtered_day_df.sort_values(by='dteday').tail(30)

        # Calculate daily changes
        recent_data['daily_change'] = recent_data['cnt'].diff()

        plt.figure(figsize=(12, 6))

        # Line plot for rental counts
        line_plot, = plt.plot(recent_data['dteday'], recent_data['cnt'], marker='o', label='Daily Rental Counts', color='blue', alpha=0.7)

        # Highlight increasing and decreasing trends
        for i in range(1, len(recent_data)):
            if recent_data['daily_change'].iloc[i] > 0:
                plt.scatter(recent_data['dteday'].iloc[i], recent_data['cnt'].iloc[i], color='green', zorder=5)
                if abs(recent_data['daily_change'].iloc[i]) > 500:  # Adjust the threshold as needed
                    plt.text(recent_data['dteday'].iloc[i], recent_data['cnt'].iloc[i] + (recent_data['cnt'].iloc[i] * 0.04), 
                            f"{int(recent_data['daily_change'].iloc[i])}", color='green', ha='center', fontsize=8,
                            bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.5'))  # Background color with border
                
            elif recent_data['daily_change'].iloc[i] < 0:
                plt.scatter(recent_data['dteday'].iloc[i], recent_data['cnt'].iloc[i], color='red', zorder=5)
                if abs(recent_data['daily_change'].iloc[i]) > 500:  # Adjust the threshold as needed
                    plt.text(recent_data['dteday'].iloc[i], recent_data['cnt'].iloc[i] + (recent_data['cnt'].iloc[i] * 0.09), 
                            f"{int(recent_data['daily_change'].iloc[i])}", color='red', ha='center', fontsize=8,
                            bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))  # Background color with border

        # Create dummy scatter plots for the legend
        scatter_increase = plt.scatter([], [], color='green', label='Increase')
        scatter_decrease = plt.scatter([], [], color='red', label='Decrease')

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

        st.pyplot(plt)
    except:
        st.error("An error occurred while generating the plot.")

    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Frequency
    # ++++++++++++++++++++++++++++++++++++++++++++++++
    st.subheader("Frequency")
    st.markdown("Seasonal and weekly pattern in the frequency of usage of the bikes.")
    # Add a 'season' label for better visualization
    try:
        season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
        filtered_day_df['season_label'] = filtered_day_df['season'].map(season_map)

        # Add a 'day' label for better visualization
        day_map = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 0: 'Sunday'}
        filtered_day_df['day_label'] = filtered_day_df['weekday'].map(day_map)

        # Aggregate data to calculate average rentals by season and weekday
        seasonal_frequency = filtered_day_df.groupby('season_label')['cnt'].mean().reset_index().sort_values('cnt', ascending=False)
        weekday_frequency = filtered_day_df.groupby(['day_label', 'weekday'])['cnt'].mean().reset_index().sort_values('weekday')

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 8))

        # Plot Frequency - Season
        colors = ["#ff0000", "#ffb5b5", "#ffb5b5", "#ffb5b5"]

        sns.barplot(x='season_label', y='cnt', data=seasonal_frequency, palette=colors, hue='season_label', ax=ax[0])
        ax[0].set_title('Average Rental Count by Season', fontsize=16, fontweight='bold')
        ax[0].set_xlabel('Season', fontsize=14, fontweight='bold')
        ax[0].set_ylabel('Average Rental Count', fontsize=14, fontweight='bold')

        # Plot Frequency - Weekday
        sns.barplot(x='day_label', y='cnt', data=weekday_frequency, palette='coolwarm', hue='day_label', ax=ax[1])
        ax[1].set_title('Average Rental Count by Weekday', fontsize=16, fontweight='bold')
        ax[1].set_xlabel('Weekday', fontsize=14, fontweight='bold')
        ax[1].set_ylabel('Average Rental Count', fontsize=14, fontweight='bold')

        st.pyplot(fig)
    except NameError:
        st.error("An error occurred while generating the plot.")

    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Monetary
    # ++++++++++++++++++++++++++++++++++++++++++++++++
    st.subheader("Monetary")
    st.markdown("Sensitivity of casual vs registered users to weather conditions")
    try:
        # Analyze sensitivity of casual vs registered users to weather conditions
        weather_sensitivity = filtered_day_df.groupby('weathersit')[['casual', 'registered']].mean().reset_index()
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

        # Plotting combined bar chart and line plots for casual and registered users
        plt.figure(figsize=(12, 8))

        sns.barplot(x=weather_name, y=registered_count, data=weather_sensitivity, color='silver', alpha=0.7, label='Registered')
        sns.barplot(x=weather_name, y=casual_count, data=weather_sensitivity, color='lightsteelblue', label='Casual')

        # Adding percentage change lines
        plt.plot(weather_name, registered_count, marker='o', color='red', label='Registered % Change', linestyle='--')
        plt.plot(weather_name, casual_count, marker='o', color='darkmagenta', label='Casual % Change', linestyle='--')

        # Customizing the plot
        plt.title('Sensitivity to Weather Conditions (Casual vs Registered)', fontsize=16, weight='bold')
        plt.xlabel('Weather Condition', fontsize=12)
        plt.ylabel('Counts / Percentage Change', fontsize=12)
        plt.legend(title='Legend', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Annotating the bar and line plots
        for i in range(length):
            # Annotating percentage changes
            plt.text(i, registered_count[i] + (registered_count[i] * 0.03), f"{percent_change_registered[i]:.1f}%", ha='center', color='red', fontsize=12)
            plt.text(i, casual_count[i] + (casual_count[i] * 0.15), f"{percent_change_casual[i]:.1f}%", ha='center', color='darkmagenta', fontsize=12)

        # Show the plot
        plt.tight_layout()
        st.pyplot(plt)
    except NameError:
        st.error("An error occurred while generating the plot.")
