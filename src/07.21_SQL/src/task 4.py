import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd

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

    # Define SQL queries to fetch data from each table
    queries = {
        'Shops': "SELECT * FROM Shops",
        'Products': "SELECT * FROM Products",
        'Customers': "SELECT * FROM Customers",
        'Orders': "SELECT * FROM Orders"
    }

    # Dictionary to hold DataFrames
    dataframes = {}

    # Fetch data and store it in DataFrames
    for table, query in queries.items():
        # Use pandas to read the SQL query into a DataFrame
        df = pd.read_sql_query(query, connection)
        dataframes[table] = df
        print(f"Data from {table}:")
        print(df)
        print()

    # Optionally, you can save the DataFrames to CSV files
    for table, df in dataframes.items():
        df.to_csv(f"{table}.csv", index=False)
        print(f"{table} data saved to {table}.csv")

    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("PostgreSQL connection closed")

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)
