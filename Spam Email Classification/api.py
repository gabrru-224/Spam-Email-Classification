from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

# Tell Flask where templates & static files are
app = Flask(__name__, template_folder="templates", static_folder="static")

# Enable CORS if you later access the API from other origins
CORS(app)

# ---------- Model Loading ---------- #

def load_model_and_vectorizer():
    with open('naive_bayes_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()


# ---------- Core Prediction Logic ---------- #

def get_spam_index(classes):
    """
    Try to robustly determine which column of predict_proba is 'spam'.
    Works for:
      - labels 'spam' / 'ham'
      - labels 0 / 1 (with 1 = spam)
    """
    spam_index = None

    # 1) Try by name (e.g. 'spam')
    for i, c in enumerate(classes):
        c_str = str(c).lower()
        if c_str == "spam":
            spam_index = i
            break

    # 2) Try common numeric encoding (1 = spam)
    if spam_index is None:
        for i, c in enumerate(classes):
            if c == 1:
                spam_index = i
                break

    # 3) Fallback: for binary 0/1, the larger label is usually the positive class
    if spam_index is None:
        if len(classes) == 2 and np.issubdtype(classes.dtype, np.number):
            spam_index = int(np.argmax(classes))  # index of 1 in [0, 1]
        else:
            # If we reach here, we don't know which class is spam
            raise ValueError(f"Cannot infer spam class from model classes_: {classes}")

    return spam_index


def predict_spam(message, threshold=0.5):
    """
    Return (label, spam_probability) for a given message string.
    """
    transformed = vectorizer.transform([message])
    proba = model.predict_proba(transformed)[0]
    classes = model.classes_

    # Find which column corresponds to spam
    spam_index = get_spam_index(classes)

    spam_prob = float(proba[spam_index])
    label = "Spam" if spam_prob >= threshold else "Ham"
    return label, spam_prob


# ---------- Routes ---------- #

@app.route("/", methods=["GET"])
def home():
    # This will load templates/index.html
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "status": "error",
            "error": "Invalid or missing JSON in request body."
        }), 400

    message = data.get("message", "")

    if not isinstance(message, str) or message.strip() == "":
        return jsonify({
            "status": "error",
            "error": "Field 'message' must be a non-empty string."
        }), 400

    try:
        label, prob = predict_spam(message)
    except Exception as e:
        # If something goes wrong in prediction (e.g. cannot infer spam class)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

    return jsonify({
        "status": "success",
        "label": label,
        "probability": prob
    })


if __name__ == "__main__":
    app.run(debug=True)
