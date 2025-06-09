import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('selangor_data.csv')
df['location'] = df['location'].astype(str).str.strip()  # Key: remove spaces and convert to string

furn_map = {
    'Unfurnished': 0,
    'Partially Furnished': 1,
    'Fully Furnished': 2,
    0: 0, 1: 1, 2: 2
}

# Add reverse mapping
furn_map_reverse = {
    0: 'Unfurnished',
    1: 'Partially Furnished',
    2: 'Fully Furnished'
}

df['furnished'] = df['furnished'].map(furn_map)

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def recommend_location(monthly_rent, rooms, furnished, topk=3):
    # Create feature vector for the query
    query_vector = np.array([monthly_rent, rooms, furn_map.get(furnished, 0)])
    
    # Normalize the features
    scaler = MinMaxScaler()
    features = df[['monthly_rent', 'rooms', 'furnished']].values
    normalized_features = scaler.fit_transform(features)
    normalized_query = scaler.transform(query_vector.reshape(1, -1))[0]
    
    # Calculate similarities
    similarities = []
    for feature_vector in normalized_features:
        similarity = cosine_similarity(normalized_query, feature_vector)
        similarities.append(similarity)
    
    # Add similarities to dataframe
    df['similarity'] = similarities
    
    # Get top k recommendations
    best = df.sort_values('similarity', ascending=False).groupby('location').first().sort_values('similarity', ascending=False).head(topk)
    best = best.reset_index()
    
    # Ensure all required columns exist
    required_columns = ['prop_name', 'location', 'monthly_rent', 'size', 'rooms', 'bathroom', 'parking', 'furnished', 'facilities','additional_facilities', 'similarity']
    missing_columns = [col for col in required_columns if col not in best.columns]
    if missing_columns:
        print(f"Warning: Following columns are missing: {missing_columns}")
        print(f"Available columns: {best.columns.tolist()}")
        return None
    
    result = best[required_columns].copy()
    # Convert furnished numbers to text descriptions
    result['furnished'] = result['furnished'].map(furn_map_reverse)
    int_cols = ['monthly_rent', 'size', 'rooms', 'bathroom', 'parking']
    result[int_cols] = result[int_cols].astype(int)
    return result

if __name__ == "__main__":
    monthly_rent = int(input("Please enter your target monthly rent: "))
    rooms = int(input("Please enter number of bedrooms: "))
    furnished = input("Please enter furnished status (Unfurnished/Partially Furnished/Fully Furnished): ")

    result = recommend_location(monthly_rent, rooms, furnished, topk=5)
    if result is not None:
        print("\nRecommended locations:")
        print(result[['prop_name', 'location', 'monthly_rent', 'size', 'rooms', 'bathroom', 'parking', 'furnished', 'facilities','additional_facilities']])
    else:
        print("Unable to generate recommendations. Please check the data format.")