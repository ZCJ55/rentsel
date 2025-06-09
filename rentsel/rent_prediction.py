import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Read data
df = pd.read_csv('selangor_data.csv')

# 2. Furnished feature mapping
furn_map = {
    'Unfurnished': 0,
    'Partially Furnished': 1,
    'Fully Furnished': 2,
    0: 0, 1: 1, 2: 2
}
df['furnished'] = df['furnished'].map(furn_map)

# 3. Location one-hot encoding
location_dummies = pd.get_dummies(df['location'], prefix='location')
df = pd.concat([df, location_dummies], axis=1)

# 4. Features and target
features = ['rooms', 'parking', 'bathroom', 'size', 'furnished'] + list(location_dummies.columns)
X = df[features]
y = df['monthly_rent']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Save model and scaler
joblib.dump(model, 'rent_model.joblib')
joblib.dump(scaler, 'rent_scaler.joblib')
joblib.dump(features, 'rent_features.joblib')

# 9. Simple prediction function
def predict_rent(rooms, parking, bathroom, size, furnished, location):
    furnished_val = furn_map.get(furnished, 0)
    input_dict = {f: 0 for f in features}
    input_dict['rooms'] = rooms
    input_dict['parking'] = parking
    input_dict['bathroom'] = bathroom
    input_dict['size'] = size
    input_dict['furnished'] = furnished_val
    loc_col = f'location_{location}'
    if loc_col in input_dict:
        input_dict[loc_col] = 1
    
    input_df = pd.DataFrame([input_dict])
    scaler = joblib.load('rent_scaler.joblib')
    model = joblib.load('rent_model.joblib')
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    return pred

if __name__ == "__main__":
    # Interactive input
    loc = input("Please enter the location : ")
    rooms = int(input("Please enter the number of bedrooms: "))
    parking = int(input("Please enter the number of parking spots: "))
    bathroom = int(input("Please enter the number of bathrooms: "))
    size = int(input("Please enter the area size (sqft): "))
    furn = input("Please enter furnished status (Unfurnished/Partially Furnished/Fully Furnished): ")

    result = predict_rent(
        rooms=rooms,
        parking=parking,
        bathroom=bathroom,
        size=size,
        furnished=furn,
        location=loc
    )
    print(f"Predicted monthly rent: RM {result:.2f}")
