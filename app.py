from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import json
import joblib
from plotly.utils import PlotlyJSONEncoder
from chatbot_routes import chatbot_bp  

app = Flask(__name__)
app.register_blueprint(chatbot_bp)   # ✅ attach chatbot routes

# Load ML model and scaler
model = joblib.load("stress_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset (for visualization only)
try:
    df_survey = pd.read_csv("data/cleaned_dataset.csv")
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")
    df_survey = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/visualization")
def visualization():
    if df_survey is None:
        return "Error loading data file."

    df_survey_cleaned = df_survey.dropna(subset=["Occupation", "Growing_Stress"])
    df_survey_cleaned["Occupation"] = df_survey_cleaned["Occupation"].str.strip().str.title()
    df_survey_cleaned["Growing_Stress"] = df_survey_cleaned["Growing_Stress"].str.strip().str.capitalize()

    fig1 = px.histogram(
        df_survey_cleaned,
        x="Occupation",
        color="Growing_Stress",
        barmode="group",
        title="Growing Stress by Occupation",
        template="plotly_white"
    )
    fig1.update_layout(xaxis_tickangle=-45)

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


@app.route("/prediction", methods=["GET"])
def prediction():
    return render_template("prediction.html")


@app.route("/predict_result", methods=["POST"])
def predict_result():
    try:
        # Collect form inputs
        year = float(request.form["Year"])
        schizophrenia = float(request.form["Schizophrenia"])
        depression = float(request.form["Depression"])
        anxiety = float(request.form["Anxiety"])
        bipolar = float(request.form["Bipolar"])
        eating = float(request.form["Eating"])

        # Create input dataframe
        input_data = pd.DataFrame([[year, schizophrenia, depression, anxiety, bipolar, eating]],
                                  columns=["Year", "Schizophrenia", "Depression", "Anxiety", "Bipolar", "Eating"])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        label_map = {0: "Low", 1: "Medium", 2: "High"}
        prediction_label = label_map.get(prediction, "Unknown")

        return render_template("prediction.html", prediction=prediction_label)

    except Exception as e:
        return f"Error during prediction: {e}"


@app.route("/developers")
def developers():
    return render_template("developers.html")


if __name__ == "__main__":
    app.run(debug=True)
