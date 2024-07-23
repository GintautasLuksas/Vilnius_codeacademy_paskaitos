import os
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

    # Define SQL queries for inserting data
    insert_queries = {
        'Shops': """INSERT INTO Shops (ShopName, Location) VALUES (%s, %s)""",
        'Products': """INSERT INTO Products (ProductName, Price, ShopID) VALUES (%s, %s, %s)""",
        'Customers': """INSERT INTO Customers (FirstName, LastName, Email) VALUES (%s, %s, %s)""",
        'Orders': """INSERT INTO Orders (CustomerID, ProductID, Quantity) VALUES (%s, %s, %s)"""
    }

    # Define data to be inserted into each table
    data = {
        'Shops': ("Book Store", "4th Floor"),
        'Products': ("Novel", 15.99, 1),  # Assumes ShopID 1 exists
        'Customers': ("David", "Wilson", "david.wilson@example.com"),
        'Orders': (3, 2, 1)  # Assumes CustomerID 3 and ProductID 2 exist
    }

    # Execute each insert query with the corresponding data
    for table, query in insert_queries.items():
        cursor.execute(query, data[table])

    # Commit the transaction to save changes to the database
    connection.commit()

    print("Records successfully added to each table!")

    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("PostgreSQL connection closed")

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)
