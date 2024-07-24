import os
# pip install python-dotenv
from dotenv import load_dotenv
import psycopg2


load_dotenv()

try:
    connection = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USERNAME'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('HOST'),
        port=os.getenv('PORT')
    )


    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    # Execute a test query
    cursor.execute("SELECT version();")

    # Fetch and print the result of the query
    record = cursor.fetchone()
    print("You are connected to - ", record, "\n")

#     INSERT
    # Executing an INSERT query
    # insert_query = """INSERT INTO employees (name, age, department) VALUES (%s, %s, %s)"""
    # record_to_insert = ("name", "age", "department")
    # cursor.execute(insert_query, record_to_insert)
    #
    # # Commit the transaction
    # connection.commit()

    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)