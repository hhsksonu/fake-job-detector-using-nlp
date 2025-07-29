# 🕵️ Fake Job Post Detector – Streamlit App

This is the web application interface for the **Fake Job Detector** project. It uses Natural Language Processing (TF-IDF) with an XGBoost classifier to determine whether a job posting is **FAKE** or **REAL**.

---

## 📦 Features

- Paste any job posting and get real-time predictions
- Confidence percentage displayed
- SHAP explanation to visualize important features influencing the decision
- Built with Streamlit, powered by XGBoost and TF-IDF

---

## 📁 File Structure

- `app.py` – Streamlit app script for local or cloud deployment
- `app.ipynb` – Colab notebook alternative (if `app.py` fails locally)
- `xgb_fake_job_model.pkl` – Pretrained XGBoost model
- `tfidf_vectorizer.pkl` – TF-IDF vectorizer used for transforming job posts
- `requirements.txt` – Required libraries for local setup

---

## 🖥️ Run the App Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
