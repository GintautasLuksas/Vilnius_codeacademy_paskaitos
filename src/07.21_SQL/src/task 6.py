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

    # Define SQL update queries
    update_queries = {
        'Update Order Quantity': """
            UPDATE Orders
            SET Quantity = %s
            WHERE OrderID = %s
        """,
        'Update Product Price': """
            UPDATE Products
            SET Price = %s
            WHERE ProductID = %s
        """
    }

    # Example updates
    updates = [
        # Update quantity for an order
        {'query': update_queries['Update Order Quantity'], 'params': (5, 1)},  # Set Quantity to 5 where OrderID = 1

        # Update price for a product
        {'query': update_queries['Update Product Price'], 'params': (749.99, 2)}
        # Set Price to 749.99 where ProductID = 2
    ]

    # Execute each update query
    for update in updates:
        cursor.execute(update['query'], update['params'])
        print(f"Update executed: {update}")

    # Commit the transaction to apply changes
    connection.commit()

    print("Data updated successfully!")

    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("PostgreSQL connection closed")

except (Exception, psycopg2.Error) as error:
    print("Error while updating data in PostgreSQL", error)
