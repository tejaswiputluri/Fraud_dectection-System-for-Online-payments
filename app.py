from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure data is received in JSON format
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON format"}), 400

        # Extract transaction details
        transaction_features = np.array([
            [
                data["type"], 
                data["amount"], 
                data["oldbalanceOrg"], 
                data["newbalanceOrig"], 
                data["oldbalanceDest"], 
                data["newbalanceDest"]
            ]
        ])

        # Make predictions
        prediction = model.predict(transaction_features)[0]
        probability = model.predict_proba(transaction_features)[0][1]

        return jsonify({
            "prediction": "Fraud" if prediction == 1 else "Legit",
            "fraud_probability": f"{probability * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
