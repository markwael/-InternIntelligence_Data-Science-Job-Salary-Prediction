import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Display text
st.title("💰 Data Science Job Salary Prediction")
st.write("This is a simple web app to predict salaries based on job-related features.")

# Dataset Information
st.title("📊 Dataset Information")
st.markdown("The dataset used for training the model is available at Kaggle: [Data Science Job Salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries/data).")
st.code("https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries/data", language="python")

# Load trained model with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load("C:\\AMIT\\PROJ\\intern_tak\\xgboost_salary_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure the model is saved correctly.")
        return None

model = load_model()

# App Title
st.title("💼 Salary Prediction")
st.write("Enter job details to predict the expected salary.")

# Sidebar Inputs
experience_level = st.selectbox("Experience Level", ["Entry-level", "Mid-level", "Senior-level", "Executive-level"])
company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Freelance"])
job_title = st.selectbox("Job Title", ["Data Scientist","Machine Learning Scientist","Big Data Engineer","Product Data Analyst","Machine Learning Engineer"])

# Convert categorical data into numerical values (matching training format)
experience_mapping = {"Entry-level": 1, "Mid-level": 2, "Senior-level": 3, "Executive-level": 4}
company_size_mapping = {"Small": 0, "Medium": 1, "Large": 2}
employment_mapping = {"Full-time": 0, "Part-time": 1, "Contract": 2, "Freelance": 3}
job_title_mapping = {"Data Scientist": 0, "Machine Learning Scientist": 1, "Big Data Engineer": 2, "Product Data Analyst": 3, "Machine Learning Engineer": 4}
# Create input dataframe
input_data = pd.DataFrame({
    "experience_level": [experience_mapping[experience_level]],
    "company_size": [company_size_mapping[company_size]],
    "employment_type": [employment_mapping[employment_type]],
    "job_title": [job_title_mapping[job_title]]  
})

# Ensure input data matches trained model's feature names
if model is not None:
    expected_features = model.feature_names_in_  
    input_data = input_data.reindex(columns=expected_features, fill_value=0) 

# Button to Predict
if st.button("Predict Salary 💰"):
    if model is not None:
        try:
            salary_pred = model.predict(input_data)[0]
            st.success(f"Predicted Salary: ${salary_pred:,.2f} USD")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.error("❌ Model not loaded. Check the model file path.")
