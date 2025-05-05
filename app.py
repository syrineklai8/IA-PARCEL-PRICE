from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS  # Ajouter cette ligne pour CORS

# Charger le modèle et l'encodeur
model = joblib.load('parcel_price_model.pkl')
encoder = joblib.load('category_encoder.pkl')

app = Flask(__name__)

# Appliquer CORS à toute l'application
CORS(app)

@app.route('/predict_price', methods=['POST'])
def predict_price():
    data = request.get_json()
    weight = data['weight']
    category = data['category']

    try:
        category_encoded = encoder.transform([category])[0]
    except ValueError:
        return jsonify({'error': 'Catégorie invalide'}), 400

    prediction = model.predict([[weight, category_encoded]])
    return jsonify({'predicted_price': round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
