import csv
from pymongo import MongoClient

try:
    # Connect to MongoDB
    client = MongoClient('localhost', 27017)
    
    # Create or connect to a database
    db = client['innoDeco_db']  # Replace 'my_database' with your database name
    
    # Create or connect to a collection
    collection = db['fournitures']  # Replace 'my_collection' with your collection name

    # Open CSV file and insert data into MongoDB
    with open(r'C:\Manarwork\Frontend\AI\innodeco_new.csv', 'r') as file:
        reader = csv.DictReader(file)
        
        # Insert data into MongoDB
        for row in reader:
            collection.insert_one(row)
            
    print("Data inserted successfully!")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close MongoDB connection
    client.close()
