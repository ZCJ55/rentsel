import pandas as pd

 
df = pd.read_csv('selangor_data.csv')  

# Calculate average monthly rent by location and round to 2 decimal places  
avg_rent_by_location = df.groupby('location')['monthly_rent'].mean().round(2).reset_index()  

csv_path = 'avg_rent_location.csv'  
avg_rent_by_location.to_csv(csv_path, index=False)  

print(f"The results have been saved to {csv_path}")