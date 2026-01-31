from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ✅ LOAD WITH JOBLIB (NOT PICKLE)
data = joblib.load("medical_insurance_model.pkl")
model = data["model"]
scaler = data["scaler"]
features = data["features"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    sex = request.form["sex"]
    smoker = request.form["smoker"]
    region = request.form["region"]

    input_dict = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0
    }

    df = pd.DataFrame([input_dict]).reindex(columns=features, fill_value=0)
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Insurance Cost: ₹{prediction:.2f}"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
