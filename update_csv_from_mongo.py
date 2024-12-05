import csv
import time
from pymongo import MongoClient
from bson import ObjectId

client = MongoClient('mongodb://admin:admin@localhost:27017/innoDeco_db')
db = client['innoDeco_db']
collection = db['favorites']

csv_file_path = r'C:\Manarwork\InnoDeco-Recommendation-System\Favorite_dataset\innoDecoFavorites.csv'

def add_to_csv(document, file_name=csv_file_path):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        favorite_value = 1 if document.get('favorite') else 0
        row = [document.get('userId'), document.get('productId'), favorite_value]
        writer.writerow(row)

def mark_as_synced(document_id):
    collection.update_one({'_id': ObjectId(document_id)}, {'$set': {'synced': True}})

def sync_data():
    synced_ids = set()
    try:
        with open(csv_file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    synced_ids.add(row[0])
    except FileNotFoundError:
        pass

    for document in collection.find({'synced': {'$ne': True}}):
        document_id = str(document.get('_id'))
        if document_id not in synced_ids:
            add_to_csv({
                'userId': document.get('userId'),
                'productId': document.get('productId'),
                'favorite': document.get('favorite')
            })
            mark_as_synced(document_id)
            synced_ids.add(document_id)

if __name__ == "__main__":
    while True:
        sync_data()
        time.sleep(10)
