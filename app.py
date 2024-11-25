from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('house_price_model.pkl')

# Create a Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Ensure input is in the correct format
        features = np.array([data['features']])
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
