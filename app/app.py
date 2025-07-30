import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# Load model and vectorizer
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "xgb_fake_job_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Initialize SHAP
shap.initjs()
explainer = shap.Explainer(model)

# Streamlit UI
st.title("Fake Job Post Detector (NLP + XGBoost)")
st.write("Paste a job posting below to check whether it's real or fake using an XGBoost model and SHAP explanation.")

user_input = st.text_area("Enter Job Posting Text:", height=300)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Vectorize input
        input_vector = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]

        # Show prediction result
        if prediction == 1:
            st.error(f"This job post is likely **FAKE** (Confidence: {proba[1]*100:.2f}%)")
        else:
            st.success(f"This job post is likely **REAL** (Confidence: {proba[0]*100:.2f}%)")

        # SHAP Explanation
        st.write("### SHAP Explanation")
        try:
            input_dense = input_vector.toarray()
            shap_values = explainer(input_dense)

            # Add feature names to SHAP values
            shap_values.feature_names = vectorizer.get_feature_names_out().tolist()

            # Waterfall plot
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False, max_display=15)
            st.pyplot(fig)

        except Exception as e:
            st.warning("SHAP explanation failed. Reason: " + str(e))
