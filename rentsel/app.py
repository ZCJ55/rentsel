from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import joblib
import pandas as pd
from recommend_location import recommend_location
from flask_sqlalchemy import SQLAlchemy
import datetime
from functools import wraps
import os
import io
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg to avoid tkinter error
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from mysql.connector import Error

# Initialize Flask application
app = Flask(__name__)

# 使用环境变量配置数据库
db_user = os.environ.get('DB_USER', 'ZHANG')
db_password = os.environ.get('DB_PASSWORD', '123456')
db_host = os.environ.get('DB_HOST', 'localhost')
db_name = os.environ.get('DB_NAME', 'rentsel_db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.environ.get('SECRET_KEY', 'a_very_secret_key_for_dev')
db = SQLAlchemy(app)

# Global request handler to check authentication
@app.before_request
def before_request():
    public_routes = ['login', 'register', 'check_email', 'static', 'about']
    if request.endpoint and request.endpoint not in public_routes and 'user_email' not in session:
        return redirect(url_for('login'))

# User model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def check_password(self, password):
        return self.password == password

# Decorator to protect routes that require authentication
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Load machine learning models and features
model = joblib.load('rent_model.joblib')
scaler = joblib.load('rent_scaler.joblib')
features = joblib.load('rent_features.joblib')

# Mapping for furnishing status
furn_map = {
    'Unfurnished': 0,
    'Partially Furnished': 1,
    'Fully Furnished': 2
}

# Database model for rental forecast history
class RentalForecastHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    forecast_value = db.Column(db.Float)
    forecast_date = db.Column(db.String(50))
    location = db.Column(db.String(100))
    rooms = db.Column(db.Integer)
    parking = db.Column(db.Integer)
    bathroom = db.Column(db.Integer)
    size = db.Column(db.Integer)
    furnished = db.Column(db.String(50))

# Database model for listing recommendation history
class ListingRecommendHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    listing_title = db.Column(db.String(100))
    recommend_date = db.Column(db.String(50))
    location = db.Column(db.String(100))
    monthly_rent = db.Column(db.Float)
    rooms = db.Column(db.Integer)
    size = db.Column(db.Integer)
    furnished = db.Column(db.String(50))

# Helper function to get available locations
def get_locations():
    return [f.replace('location_', '') for f in features if f.startswith('location_')]

# Main route for rental price prediction
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    prediction = None
    locations = get_locations()
    user_email = session.get('user_email')
    if request.method == 'POST':
        try:
            # Get form data
            loc = request.form['location']
            rooms = int(request.form['rooms'])
            parking = int(request.form['parking'])
            bathroom = int(request.form['bathroom'])
            size = int(request.form['size'])
            furnished = request.form['furnished']
            
            # Prepare input for prediction
            input_dict = {f: 0 for f in features}
            input_dict['rooms'] = rooms
            input_dict['parking'] = parking
            input_dict['bathroom'] = bathroom
            input_dict['size'] = size
            input_dict['furnished'] = furn_map.get(furnished, 0)
            loc_col = f'location_{loc}'
            if loc_col in input_dict:
                input_dict[loc_col] = 1
            
            # Make prediction
            input_df = pd.DataFrame([input_dict])
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            prediction = f"RM {pred:.2f} /month"
            
            # Save prediction to history
            db.session.add(RentalForecastHistory(
                forecast_value=pred,
                forecast_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                location=loc,
                rooms=rooms,
                parking=parking,
                bathroom=bathroom,
                size=size,
                furnished=furnished
            ))
            db.session.commit()
            
            return render_template('index.html', 
                               locations=locations, 
                               prediction=prediction, 
                               user_email=user_email,
                               loc=loc, rooms=rooms, parking=parking, 
                               bathroom=bathroom, size=size, furnished=furnished)
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('index.html', locations=locations, prediction=prediction, user_email=user_email)

# Route for property recommendations
@app.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    user_email = session.get('user_email')
    monthly_rent = None
    rooms = None
    furnished = None
    recommendations = None

    if request.method == 'POST':
        try:
            # Get user preferences
            monthly_rent = float(request.form['monthly_rent'])
            rooms = float(request.form['rooms'])
            furnished = request.form['furnished']
            
            # Get recommendations
            recommendations = recommend_location(monthly_rent, rooms, furnished, topk=5)
            if recommendations is not None:
                recommendations = recommendations.to_dict('records')
                # Save first recommendation to history
                if len(recommendations) > 0:
                    rec = recommendations[0]
                    db.session.add(ListingRecommendHistory(
                        listing_title=f"{rec['location']} ",
                        recommend_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        location=rec['location'],
                        monthly_rent=rec['monthly_rent'],
                        size=rec['size'],
                        rooms=rec['rooms'],
                        furnished=rec['furnished']
                    ))
                    db.session.commit()
        except Exception as e:
            recommendations = None
            print(f"Error in recommendation: {str(e)}")
    elif request.method == 'GET':
        # Handle recommendations from history
        monthly_rent_param = request.args.get('monthly_rent')
        rooms_param = request.args.get('rooms')
        furnished_param = request.args.get('furnished')

        if monthly_rent_param and rooms_param and furnished_param:
            try:
                monthly_rent = float(monthly_rent_param)
                rooms = float(rooms_param)
                furnished = furnished_param
                
                recommendations = recommend_location(monthly_rent, rooms, furnished, topk=5)
                if recommendations is not None:
                    recommendations = recommendations.to_dict('records')
            except ValueError:
                print("Invalid parameters received in GET request.")

    return render_template('recommend.html', recommendations=recommendations, user_email=user_email, 
                           monthly_rent=monthly_rent, rooms=rooms, furnished=furnished)

# Route to display prediction and recommendation history
@app.route('/history')
@login_required
def history():
    user_email = session.get('user_email')
    rental_history = RentalForecastHistory.query.order_by(RentalForecastHistory.id.desc()).limit(20).all()
    listing_history = ListingRecommendHistory.query.order_by(ListingRecommendHistory.id.desc()).limit(20).all()
    return render_template('history.html', rental_history=rental_history, listing_history=listing_history, user_email=user_email)

# Route to delete rental prediction history
@app.route('/delete_rent/<int:rid>', methods=['POST'])
@login_required
def delete_rent(rid):
    record = RentalForecastHistory.query.get(rid)
    if record:
        db.session.delete(record)
        db.session.commit()
    return redirect(url_for('history'))

# Route to delete listing recommendation history
@app.route('/delete_listing/<int:lid>', methods=['POST'])
@login_required
def delete_listing(lid):
    record = ListingRecommendHistory.query.get(lid)
    if record:
        db.session.delete(record)
        db.session.commit()
    return redirect(url_for('history'))

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['user_email'] = user.email
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid email or password')
    
    if 'user_email' in session:
        return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    email = request.form.get('regEmail')
    password = request.form.get('regPassword')
    confirm_password = request.form.get('confirmPassword')

    if password != confirm_password:
        return redirect(url_for('login', error='Passwords do not match!'))

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return redirect(url_for('login', error='This email is already registered.'))

    new_user = User(email=email)
    new_user.password = password
    try:
        db.session.add(new_user)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error during user registration: {e}")
        return redirect(url_for('login', error='Registration failed due to a server error.'))
    
    return redirect(url_for('login', success='Registration successful! Please login.'))

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('login'))

# API endpoint to check email availability
@app.route('/check_email', methods=['GET'])
def check_email():
    email = request.args.get('email')
    user = User.query.filter_by(email=email).first()
    if user:
        return jsonify(is_taken=True)
    return jsonify(is_taken=False)

# About page route
@app.route('/about')
def about():
    return render_template('about.html')

# Function to generate rental price visualization
def generate_plot():
    plt.rcParams['font.size'] = 18

    try:
        # Connect to database
        connection = mysql.connector.connect(
            host='localhost',
            database='rentsel_db',
            user='ZHANG',
            password='123456'
        )

        if connection.is_connected():
            cursor = connection.cursor()
            # Get locations from historical data
            cursor.execute("SELECT DISTINCT location FROM rental_forecast_history")
            historical_locations = [row[0] for row in cursor.fetchall()]

            df = pd.read_csv('avg_rent_location.csv')
            df = df[df['location'].isin(historical_locations)]

            # Create visualization
            plt.figure(figsize=(16, 8))
            ax = sns.barplot(x='location', y='monthly_rent', data=df)

            # Add value labels
            for i, v in enumerate(df['monthly_rent']):
                ax.text(i, v, f'RM {v:.2f}', ha='center', va='bottom', fontsize=16)

            plt.title('Average Monthly Rent by Location', fontsize=24)
            plt.xlabel('Location', fontsize=20)
            plt.ylabel('Monthly Rent (RM)', fontsize=20)
            plt.ylim(0, 5000)
            plt.xticks(rotation=45, ha='right', fontsize=18)
            plt.tight_layout()

        
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
            img.seek(0)
            plt.close()

            return img

    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed")

# Routes for visualization
@app.route('/visualization')
@login_required
def visualization():
    user_email = session.get('user_email')
    return render_template('visualization.html', user_email=user_email)

@app.route('/plot')
@login_required
def plot():
    img = generate_plot()
    return send_file(img, mimetype='image/png')

# Application entry point
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)