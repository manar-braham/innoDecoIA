import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Charger le modèle
model = load_model(r"C:\Manarwork\train_Model\models\model_vgg16_2.keras")

class_names = ['bar-stool', 'bathroom-accessorie', 'bed', 'bookcase', 
               'cabinet', 'candleholder', 'candles-and-home-fragrance', 'chair', 
               'chest-of-drawer', 'clock', 'decoration', 'footstool', 'kitchen-accessorie',
               'knobs-and-handle', 'light', 'living-room-accessorie', 'mirror', 'organiser',
               'outdoor-cooking', 'plant', 'sofa', 'table', 'vases-and-bowl', 'wardrobe']

def preprocess_image(image_path, target_size=(100, 100)):
    img = cv2.imread(image_path)
    if img is None:
        print("Erreur: l'image n'a pas pu être chargée.")
        sys.exit(1)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR à RGB
    img = img.astype("float") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Utilisation: python predict_image.py chemin_vers_image")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print("Le chemin spécifié n'est pas un fichier valide.")
        sys.exit(1)

    # Prétraiter l'image
    preprocessed_image = preprocess_image(image_path)

    try:
        # Faire la prédiction
        prediction = model.predict(preprocessed_image)
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        sys.exit(1)

    # Obtenir le nom de la classe prédite
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]

    # Afficher la prédiction uniquement
    print(predicted_class)
