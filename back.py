from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained ML model
model = joblib.load("phishing_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        if not isinstance(data, dict):
            return jsonify({"error": "Send data as JSON dictionary"}), 400

        # Data must contain all numeric columns from training dataset
        features = pd.DataFrame([data])

        # Predict
        prediction = model.predict(features)[0]
        confidence = float(np.max(model.predict_proba(features)))

        result = "Legitimate" if prediction == 0 else "Phishing"

        return jsonify({
            "prediction": result,
            "confidence_score": round(confidence, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Phishing Detection Backend Running Successfully!"

if __name__ == '__main__':
    app.run(debug=True)
