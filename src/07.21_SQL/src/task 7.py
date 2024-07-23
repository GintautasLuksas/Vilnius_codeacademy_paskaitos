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

    # Define the SQL delete query
    delete_query = """
        DELETE FROM Orders
        WHERE OrderID = %s
    """

    # Specify the OrderID to delete
    order_id_to_delete = 1  # Replace with the ID of the order you want to delete

    # Execute the delete query
    cursor.execute(delete_query, (order_id_to_delete,))

    # Commit the transaction to apply changes
    connection.commit()

    print(f"Order with OrderID {order_id_to_delete} deleted successfully!")

    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("PostgreSQL connection closed")

except (Exception, psycopg2.Error) as error:
    print("Error while deleting data from PostgreSQL", error)
