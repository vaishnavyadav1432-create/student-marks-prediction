import streamlit as st
import pandas as pd
import joblib

model = joblib.load("student_marks_model.pkl")

st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Marks Prediction System")
st.subheader("🤖 Machine Learning Based Prediction")

st.write("Enter student details to predict final marks.")

study_hours = st.number_input(
    "📚 Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

attendance = st.number_input(
    "📅 Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0
)

previous_marks = st.number_input(
    "📝 Previous Marks",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

assignment_score = st.number_input(
    "📊 Assignment Score",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

if st.button("🔮 Predict Final Marks"):
    new_student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks],
        "Assignment_Score": [assignment_score]
    })

    prediction = model.predict(new_student)[0]
    prediction = max(0, min(100, prediction))

    if prediction >= 75:
        performance = "Excellent 🌟"
    elif prediction >= 60:
        performance = "Good 👍"
    elif prediction >= 40:
        performance = "Average 📚"
    else:
        performance = "Needs Improvement 💪"

    st.success(f"🎯 Predicted Final Marks: {prediction:.2f} / 100")
    st.info(f"📊 Performance: {performance}")

st.markdown("---")
st.markdown("""
### 📌 Project Information

**Algorithm:** Linear Regression

**Technologies:** Python, Pandas, Scikit-learn, Streamlit

**Purpose:** Student Final Marks Prediction
""")
