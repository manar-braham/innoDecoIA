# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
import pickle
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Fonction pour extraire les caractéristiques d'une image à partir d'une URL
def extract_img_features_from_url(img_url, model, session):
    try:
        response = session.get(img_url, timeout=10)
        response.raise_for_status()  # Lève une HTTPError en cas de mauvais statut
        img = image.load_img(BytesIO(response.content), target_size=(224, 224))
        img_array = image.img_to_array(img)
        expand_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expand_img)
        result_to_resnet = model.predict(preprocessed_img)
        flatten_result = result_to_resnet.flatten()
        result_normalized = flatten_result / norm(flatten_result)
        return result_normalized
    except requests.exceptions.RequestException as e:
        print(f"Échec de la requête pour l'URL {img_url}: {e}")
        return None

# Configurer la stratégie de réessai
retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("http://", adapter)
session.mount("https://", adapter)

# Chemins vers le fichier CSV et le fichier de caractéristiques
csv_file = 'C:/Manarwork/train_Model/innodeco_clean.csv'
features_path = 'C:/Manarwork/train_Model/features.pkl'

# Modèle ResNet50 pré-entraîné
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Lire le fichier CSV
data = pd.read_csv(csv_file)

features_list = []
img_files_list = []

for index, row in data.iterrows():
    img_url = row['Product_URL']
    img_name = row['Product_URL']
    img_files_list.append(img_name)
    
    # Extraire les caractéristiques de l'image à partir de l'URL
    features = extract_img_features_from_url(img_url, model, session)
    if features is not None:
        features_list.append(features)
    else:
        print(f"Échec de l'extraction des caractéristiques pour {img_url}")

# Convertir les listes en arrays numpy
features_list = np.array(features_list)

# Enregistrer les caractéristiques et les noms des images dans un fichier
with open(features_path, 'wb') as f:
    pickle.dump((features_list, img_files_list), f)

print(f"Extraction des caractéristiques terminée et enregistrées dans {features_path}")