from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

# Initialize Flask app
app = Flask(__name__)

# Load models and scaler
scaler = joblib.load("model/scaler.pkl")
ann_model = load_model("model/fake_profile_model.h5")
xgb_model = joblib.load("model/xgb_model.pkl")  # Save XGBoost separately

# Select the best model (hardcoded for now, can be auto-selected based on accuracy)
BEST_MODEL = "XGBoost"  # Change to "ANN" if ANN is better

# Function to predict using the best model
def predict_fake_profile(data):
    # Convert input to numpy array
    user_input = np.array([data])

    # Scale input
    user_input_scaled = scaler.transform(user_input)

    # Predict using the best model
    if BEST_MODEL == "XGBoost":
        prediction = xgb_model.predict(user_input)
    else:
        prediction = np.argmax(ann_model.predict(user_input_scaled), axis=1)

    return "FAKE Profile! 🚨 Ban the ID as soon as possible!" if prediction[0] == 1 else "✅ NO Fake Profile. The account is safe!"

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        data = [
            int(request.form["profile_pic"]),
            float(request.form["nums_length_username"]),
            int(request.form["fullname_words"]),
            float(request.form["nums_length_fullname"]),
            int(request.form["name_equals_username"]),
            int(request.form["description_length"]),
            int(request.form["external_url"]),
            int(request.form["private"]),
            int(request.form["num_posts"]),
            int(request.form["num_followers"]),
            int(request.form["num_follows"]),
        ]

        # Get prediction
        result = predict_fake_profile(data)

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
