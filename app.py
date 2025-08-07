from flask import Flask, render_template, request
import pickle, joblib

app = Flask(__name__)

# Load your trained pipeline (model + vectorizer)
with open("fake_news_detector.pkl", "rb") as file:
    pipeline = joblib.load(file)

# Step 1: Define a simple fact-check function
def simple_fact_check(text):
    # List of known false rumors
    known_false_claims = [
        "trump is dead",
        "donald trump died",
        "joe biden dead",
        "modi died",
        "elon musk is dead",
        "bill gates is dead",
        "mark zuckerberg died",
        "obama died",
        "putin is dead",
        "celebrity death hoax"
    ]
    text_lower = text.lower()
    for claim in known_false_claims:
        if claim in text_lower:
            return False  # Fake News Detected
    return None  # Unknown → Let ML model decide

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']

    # Step 2: Check against simple fact-check list first
    fact_check_result = simple_fact_check(news_text)
    if fact_check_result is False:
        result = "Fake News ❌ (Known false rumor)"
    else:
        # Step 3: Fallback to ML pipeline
        prediction = pipeline.predict([news_text])[0]
        result = "Fake News ❌" if prediction == 0 else "Real News ✅"

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)