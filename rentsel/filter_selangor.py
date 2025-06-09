import pandas as pd
import re

# Read the original CSV file
df = pd.read_csv('mudah-apartment-kl-selangor.csv')

# Filter data for Selangor region (Keeping all data for now based on previous requests)
df_selangor = df

# Keep only specified columns
columns_to_keep = ['prop_name', 'monthly_rent', 'location', 'rooms', 'parking', 'bathroom', 'size', 'furnished', 'facilities', 'additional_facilities', 'region']
df_selangor = df_selangor[columns_to_keep]

# Function to clean and extract a single number (for rooms, parking, bathroom)
# Returns None if 'More than 10' is found, to allow dropping these rows
def clean_and_extract_single_number(value_str):
    if pd.isna(value_str):
        return None
    s = str(value_str).strip()
    if 'more than 10' in s.lower(): # Explicitly check for 'More than 10' and mark for dropping
        return None
    numbers = re.findall(r'\d+', s)
    if numbers:
        return int(numbers[0])
    return None

# Function to extract full numbers (for monthly_rent, size, handles currency and commas)
def extract_full_number(value_str):
    if pd.isna(value_str):
        return None
    cleaned_str = str(value_str).replace('RM', '').replace(',', '').strip()
    numbers = re.findall(r'\d+', cleaned_str)
    if numbers:
        return int(''.join(numbers))
    return None

# Apply processing functions
df_selangor['monthly_rent'] = df_selangor['monthly_rent'].apply(extract_full_number)
df_selangor['size'] = df_selangor['size'].apply(extract_full_number)
df_selangor['rooms'] = df_selangor['rooms'].apply(clean_and_extract_single_number)
df_selangor['parking'] = df_selangor['parking'].apply(clean_and_extract_single_number)
df_selangor['bathroom'] = df_selangor['bathroom'].apply(clean_and_extract_single_number)

# Function to remove numerical outliers based on reasonable bounds
def remove_numerical_outliers(df):
    print("\nNumerical outliers before removal:")
    print(df.describe())

    # Define reasonable bounds for each numerical feature
    bounds = {
        'monthly_rent': (100, 50000), # Example: RM 100 to RM 50,000
        'size': (100, 10000),         # Example: 100 sqft to 10,000 sqft
        'rooms': (1, 10),             # Example: 1 to 10 rooms
        'parking': (0, 5),            # Example: 0 to 5 parking spots
        'bathroom': (1, 10)           # Example: 1 to 10 bathrooms
    }

    initial_rows = len(df)
    for col, (lower, upper) in bounds.items():
        if col in df.columns:
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows due to numerical outliers.")
    else:
        print("No numerical outliers removed.")

    print("\nNumerical outliers after removal (new describe):")
    print(df.describe())
    return df

# Apply outlier removal after initial cleaning and before dropping NaNs
df_selangor = remove_numerical_outliers(df_selangor)

# Handle missing values
print("\nMissing values before cleaning:")
print(df_selangor.isnull().sum())

# Fill missing values in additional_facilities and facilities with "None"
df_selangor['additional_facilities'] = df_selangor['additional_facilities'].fillna("None")
df_selangor['facilities'] = df_selangor['facilities'].fillna("None") # Fill missing facilities

# Remove rows with missing values in critical columns (including those marked None by clean_and_extract_single_number)
df_selangor = df_selangor.dropna(subset=['prop_name', 'monthly_rent', 'location', 'rooms', 'parking', 'bathroom', 'size', 'furnished'])

print("\nMissing values after cleaning:")
print(df_selangor.isnull().sum())

# Save filtered data to new CSV file
df_selangor.to_csv('selangor_data.csv', index=False)


print("\nAll Regions Data Statistics:") # Text update, as we are processing all regions now
print(f"Total number of records: {len(df_selangor)}")
print("\nData Preview:")
print(df_selangor.head())

# Display basic information about the data
print("\nData Information:")
print(df_selangor.info()) 