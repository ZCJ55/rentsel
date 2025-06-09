import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from mysql.connector import Error

# Set font support for Chinese characters
plt.rcParams['font.size'] = 14  # Increase default font size

try:
    # Connect to MySQL database
    connection = mysql.connector.connect(
        host='localhost',
        database='rentsel_db',
        user='ZHANG',
        password='123456'
    )

    if connection.is_connected():
        cursor = connection.cursor()
        # Get unique locations from historical prediction data
        cursor.execute("SELECT DISTINCT location FROM rental_forecast_history")
        historical_locations = [row[0] for row in cursor.fetchall()]

        # Read CSV file
        df = pd.read_csv('avg_rent_location.csv')
        
        # Keep only locations that appear in historical predictions
        df = df[df['location'].isin(historical_locations)]

        # Create figure
        plt.figure(figsize=(20, 10))
        ax = sns.barplot(x='location', y='monthly_rent', data=df)

        # Add value labels on top of each bar
        for i, v in enumerate(df['monthly_rent']):
            ax.text(i, v, f'RM {v:.2f}', ha='center', va='bottom', fontsize=12)

        # Set title and labels
        plt.title('Average Monthly Rent by Location', fontsize=20)
        plt.xlabel('Location', fontsize=16)
        plt.ylabel('Monthly Rent (RM)', fontsize=16)

        # Set y-axis range
        plt.ylim(0, 5000)

        # Rotate x-axis labels to prevent overlap and increase font size
        plt.xticks(rotation=45, ha='right', fontsize=14)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plt.savefig('rent_visualization.png', dpi=300, bbox_inches='tight')

        # Display plot
        plt.show()

except Error as e:
    print(f"Error connecting to MySQL database: {e}")

finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection closed") 