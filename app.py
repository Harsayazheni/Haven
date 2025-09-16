from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import json
from plotly.utils import PlotlyJSONEncoder

# Import chatbot blueprint
from chatbot_routes import chatbot_bp  

app = Flask(__name__)
app.register_blueprint(chatbot_bp)   # ✅ attach chatbot routes

# Load dataset
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

    # Data cleaning
    df_survey_cleaned = df_survey.dropna(subset=["Occupation", "Growing_Stress"])
    df_survey_cleaned["Occupation"] = df_survey_cleaned["Occupation"].str.strip().str.title()
    df_survey_cleaned["Growing_Stress"] = df_survey_cleaned["Growing_Stress"].str.strip().str.capitalize()

    # Visualization 1
    fig1 = px.histogram(
        df_survey_cleaned,
        x="Occupation",
        color="Growing_Stress",
        barmode="group",
        title="Growing Stress by Occupation",
        template="plotly_white"
    )
    fig1.update_layout(xaxis_tickangle=-45)

    # Visualization 2
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


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/developers")
def developers():
    return render_template("developers.html")


if __name__ == "__main__":
    app.run(debug=True)
