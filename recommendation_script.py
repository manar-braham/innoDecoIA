from PIL import Image
import requests
from io import BytesIO
import sys
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
import pickle
import os

def main():
    try:
        features_path = r'C:/Manarwork/train_Model/features.pkl'
        
        # Charger les caractéristiques et les chemins d'accès aux images
        with open(features_path, 'rb') as f:
            features_list, img_files_list = pickle.load(f)

        # Charger le modèle
        model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        model.trainable = False
        model = Sequential([model, GlobalMaxPooling2D()])

        img_path_or_url = sys.argv[1]  # L'argument est maintenant le chemin ou l'URL de l'image
        distance_threshold = 0.5  # Valeur par défaut
        if len(sys.argv) > 2:
            distance_threshold = float(sys.argv[2])

        # Vérifier si le chemin fourni est une URL
        if img_path_or_url.startswith('http://') or img_path_or_url.startswith('https://'):
            # Récupérer l'image à partir de l'URL
            response = requests.get(img_path_or_url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            # Traiter comme un chemin de fichier local
            if not os.path.exists(img_path_or_url):
                raise FileNotFoundError(f"Image file not found: {img_path_or_url}")
            img = Image.open(img_path_or_url)

        # Convertir l'image en mode RGB pour assurer la compatibilité
        img = img.convert("RGB")
        
        # Redimensionner l'image pour qu'elle soit compatible avec ResNet50
        img = img.resize((224, 224))
        
        # Convertir l'image en tableau numpy
        img_array = image.img_to_array(img)
        
        # Ajouter une dimension pour obtenir la forme attendue par ResNet50
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prétraiter l'image pour la passer à ResNet50
        img_array = preprocess_input(img_array)

        # Obtenir les caractéristiques de l'image avec ResNet50
        features = model.predict(img_array)
        flatten_result = features.flatten()
        result_normalized = flatten_result / norm(flatten_result)

        # Calculer les distances entre les caractéristiques de l'image et celles de la base de données
        distances = np.linalg.norm(features_list - result_normalized, axis=1)
        
        # Sélectionner les indices des images similaires selon le seuil de distance
        indices = np.where(distances < distance_threshold)[0]
        
        # Obtenir les chemins d'accès des images recommandées
        recommended_images = [img_files_list[idx].replace('\\', '/') for idx in indices]

        # Imprimer la liste JSON des images recommandées
        json_output = json.dumps(recommended_images, ensure_ascii=False)
        print(json_output)

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
