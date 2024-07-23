import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

# Load environment variables from the .env file
load_dotenv()

# Fetch environment variables
dbname = os.getenv('DB_NAME')
user = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')
host = os.getenv('HOST')
port = os.getenv('PORT')

# Create the SQLAlchemy engine
connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(connection_string)

try:
    # Define SQL queries
    queries = {
        'Shops with Products': """
            SELECT 
                Shops.ShopName, 
                Shops.Location, 
                Products.ProductName, 
                Products.Price
            FROM 
                Shops
            LEFT JOIN 
                Products ON Shops.ShopID = Products.ShopID
        """,
        'Orders by Customer': """
            SELECT 
                Orders.OrderID, 
                Orders.OrderDate, 
                Customers.FirstName, 
                Customers.LastName, 
                Products.ProductName, 
                Orders.Quantity
            FROM 
                Orders
            JOIN 
                Customers ON Orders.CustomerID = Customers.CustomerID
            JOIN 
                Products ON Orders.ProductID = Products.ProductID
            WHERE 
                Customers.CustomerID = 1  -- Replace 1 with the specific CustomerID
        """,
        'Average Price per Shop': """
            SELECT 
                Shops.ShopName, 
                AVG(Products.Price) AS AveragePrice
            FROM 
                Shops
            JOIN 
                Products ON Shops.ShopID = Products.ShopID
            GROUP BY 
                Shops.ShopName
        """
    }

    # Execute each query and store results in DataFrames
    for title, query in queries.items():
        df = pd.read_sql_query(query, engine)
        print(f"\n{title}:")
        print(df)

    print("\nPostgreSQL connection closed")

except Exception as error:
    print("Error while connecting to PostgreSQL", error)
