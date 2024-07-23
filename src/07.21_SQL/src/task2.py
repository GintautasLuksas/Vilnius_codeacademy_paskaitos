import os
import pandas as pd
from dotenv import load_dotenv
import psycopg2

# Load environment variables from the .env file
load_dotenv()

# Fetch environment variables
dbname = os.getenv('DB_NAME')
user = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')
host = os.getenv('HOST')
port = os.getenv('PORT')

try:
    # Establish the connection to the PostgreSQL database
    connection = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )

    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    # Define queries to fetch data from each table
    queries = {
        'shops': "SELECT * FROM shops",
        'products': "SELECT * FROM products",
        'customers': "SELECT * FROM customers",
        'orders': "SELECT * FROM orders"
    }

    # Dictionary to hold DataFrames
    data_frames = {}

    # Execute each query and store results in pandas DataFrames
    for table, query in queries.items():
        cursor.execute(query)
        # Fetch all rows from the executed query
        rows = cursor.fetchall()
        # Fetch column names
        columns = [desc[0] for desc in cursor.description]
        # Create a pandas DataFrame
        df = pd.DataFrame(rows, columns=columns)
        data_frames[table] = df
        print(f"Data from {table}:")
        print(df.head(), "\n")  # Display first few rows

    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("PostgreSQL connection is closed")

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)
