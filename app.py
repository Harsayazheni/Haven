from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import json
import joblib
from plotly.utils import PlotlyJSONEncoder
from chatbot_routes import chatbot_bp   # Import chatbot blueprint

# ==============================
# Initialize Flask app
# ==============================
app = Flask(__name__)
app.register_blueprint(chatbot_bp)   # ✅ Attach chatbot routes

# ==============================
# Load ML model and encoders
# ==============================
try:
    model = joblib.load("stress_model.pkl")
    scaler = joblib.load("scaler.pkl")
    occ_encoder = joblib.load("occ_encoder.pkl")
    gender_encoder = joblib.load("gender_encoder.pkl")
    bmi_encoder = joblib.load("bmi_encoder.pkl")
except Exception as e:
    print(f"⚠️ Model or encoders not loaded: {e}")
    model, scaler, occ_encoder, gender_encoder, bmi_encoder = None, None, None, None, None

# ==============================
# Load dataset (for visualization only)
# ==============================
try:
    df_survey = pd.read_csv("data/cleaned_dataset.csv")
except FileNotFoundError as e:
    print(f"⚠️ Error loading dataset: {e}")
    df_survey = None


@app.route("/")
def index():
    """Homepage"""
    return render_template("index.html")


@app.route("/visualization")
def visualization():
    """Stress data visualization"""
    if df_survey is None:
        return "Error: Dataset not found."

    # Clean data
    df_survey_cleaned = df_survey.dropna(subset=["Occupation", "Growing_Stress"])
    df_survey_cleaned["Occupation"] = df_survey_cleaned["Occupation"].str.strip().str.title()
    df_survey_cleaned["Growing_Stress"] = df_survey_cleaned["Growing_Stress"].str.strip().str.capitalize()

    # Plot 1: Stress by occupation
    fig1 = px.histogram(
        df_survey_cleaned,
        x="Occupation",
        color="Growing_Stress",
        barmode="group",
        title="Growing Stress by Occupation",
        template="plotly_white"
    )
    fig1.update_layout(xaxis_tickangle=-45)

    # Plot 2: Stacked bar of stress levels
    occ_stress = df_survey_cleaned.groupby(['Occupation', 'Growing_Stress']).size().reset_index(name='Count')
    fig2 = px.bar(
        occ_stress,
        x="Occupation",
        y="Count",
        color="Growing_Stress",
        title="Proportion of Growing Stress by Occupation",
        barmode="stack",
        template="plotly_white"
    )
    fig2.update_layout(xaxis_tickangle=-45)

    return render_template(
        "visualization.html",
        fig1_json=json.dumps(fig1, cls=PlotlyJSONEncoder),
        fig2_json=json.dumps(fig2, cls=PlotlyJSONEncoder)
    )


# ==============================
# Prediction Routes
# ==============================
@app.route("/prediction", methods=["GET"])
def prediction():
    """Render prediction form"""
    return render_template("prediction.html")


@app.route("/predict_result", methods=["POST"])
def predict_result():
    """Handle prediction request"""
    if not model or not scaler:
        return "⚠️ Model not available. Train and upload pickle files first."

    try:
        # Collect form inputs
        gender = request.form["Gender"]
        age = float(request.form["Age"])
        occupation = request.form["Occupation"]
        sleep_duration = float(request.form["SleepDuration"])
        bmi_category = request.form["BMI"]
        heart_rate = float(request.form["HeartRate"])
        daily_steps = float(request.form["DailySteps"])
        systolic_bp = float(request.form["SystolicBP"])

        # Encode categorical values
        gender_enc = gender_encoder.transform([gender])[0]
        occ_enc = occ_encoder.transform([occupation])[0]
        bmi_enc = bmi_encoder.transform([bmi_category])[0]

        # Create input dataframe
        input_data = pd.DataFrame([[gender_enc, age, occ_enc, sleep_duration,
                                    bmi_enc, heart_rate, daily_steps, systolic_bp]],
                                  columns=["Gender", "Age", "Occupation", "Sleep Duration",
                                           "BMI Category", "Heart Rate", "Daily Steps", "Systolic BP"])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        return render_template("prediction.html", prediction=prediction)

    except Exception as e:
        return f"⚠️ Error during prediction: {e}"


@app.route("/developers")
def developers():
    """Developers page"""
    return render_template("developers.html")


if __name__ == "__main__":
    app.run(debug=True)
