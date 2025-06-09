import pymysql
from app import db, app

def init_database():
    # Create database connection
    connection = pymysql.connect(
        host='localhost',
        user='ZHANG',
        password='123456'
    )
    
    try:
        with connection.cursor() as cursor:
            # Create database
            cursor.execute("CREATE DATABASE IF NOT EXISTS rentsel_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print("Database created successfully!")
    except Exception as e:
        print(f"Error creating database: {str(e)}")
    finally:
        connection.close()

if __name__ == '__main__':
    # Create database
    init_database()
    
    # Create all tables in application context
    with app.app_context():
        db.create_all()
        print("All tables created successfully!") 