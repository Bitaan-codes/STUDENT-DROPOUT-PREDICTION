import gradio as gr
import numpy as np
import joblib

model = joblib.load("dropout_risk_prediction_model.pkl")

def predict_dropout(
        age,
        gender,
        year_of_study,
        attendance,
        study_hours,
        previous_gpa,
        backlogs,
        financial_stress,
        stress_level,
        burnout_level
):
    gender = 1 if gender =="Male" else 0

    burnout_mapping = {
        "Low" : 0,
        "Moderate" : 1,
        "High" : 2
    }
    burnout = burnout_mapping[burnout_level]

    input_data = np.array([[
         age,
        gender,
        year_of_study,
        attendance,
        study_hours,
        previous_gpa,
        backlogs,
        financial_stress,
        stress_level,
        burnout

    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1 :
        return "Student will drop out"
    else :
        return "Student is not likely to drop out"
    
student_dropout_app = gr.Interface(
    fn = predict_dropout,
    inputs = [
        gr.Number(label = "Age"),
        gr.Dropdown(choices = ["Male", "Female"], label = "Gender"),
        gr.Dropdown(choices = [1, 2, 3, 4], label = "Year of Study"),
        gr.Slider(0, 100, label = "Attendance (%)"),
        gr.Number(label = "Study Hours per Week"),
        gr.Number(label = "Previous GPA"),
        gr.Number(label = "Number of Backlogs"),
        gr.Slider(1,10,step=1, label = "Financial Stress Level"),
        gr.Slider(1,10,step=1, label = "Stress Level"),
        gr.Dropdown(choices = ["Low", "Moderate", "High"], label = "Burnout Level"),
    ],
        outputs = gr.Textbox(label = "Prediction"),
        title = "Student Dropout Risk Prediction",
        description = "Predict wether a student has High or Low dropout risk",
)

student_dropout_app.launch()

import os

port = int(os.environ.get("PORT", 7860))

student_dropout_app.launch(
    server_name="0.0.0.0",
    server_port=port
)