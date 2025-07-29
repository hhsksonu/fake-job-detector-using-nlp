
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

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Launch the Streamlit app

```bash
streamlit run app.py
```

If it runs successfully, it will open in your default web browser at:

```text
http://localhost:8501
```

---

## ⚠️ Having Trouble Running `app.py` Locally?

You can run the app using **Google Colab + ngrok**.

---

## 📄 Use `app.ipynb` in Google Colab

### Step 1: Clone the repository and go to the `app` folder  
### Step 2: Open `app.ipynb` in Google Colab  
### Step 3: Upload the following files to Colab:

- `xgb_fake_job_model.pkl`  
- `tfidf_vectorizer.pkl`  

---

### ✅ Install Required Libraries

```python
!pip install streamlit shap pyngrok
```

---

### 🔑 Add Your ngrok Authtoken

```python
!ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN
```

> Replace `YOUR_NGROK_AUTHTOKEN` with your actual ngrok token (only required once per session).

---
# 🔐 Get and Set Up Your Ngrok Authtoken

To run your Streamlit app online using [ngrok](https://ngrok.com), you'll need an **Authtoken**. Follow the steps below:

---

### Create an Ngrok Account

- Visit [https://ngrok.com](https://ngrok.com)
- Click **"Sign up"** (or log in if you already have an account).
- After signing in, you will be redirected to your **Ngrok Dashboard**.

---

### Copy Your Authtoken

- In the dashboard, go to the **"Your Authtoken"** section.
- You’ll see a command that looks like this:

```bash
ngrok config add-authtoken <YOUR_AUTHTOKEN>
```

> Replace `<YOUR_AUTHTOKEN>` with your actual token.
---
### 🚀 Run the App and Get Public URL

```python
from pyngrok import ngrok
import time
import threading

def run_streamlit():
    !streamlit run app.py

thread = threading.Thread(target=run_streamlit)
thread.start()
time.sleep(5)

public_url = ngrok.connect(8501)
print("Your app is live at:", public_url)
```

You’ll get a live Streamlit link like:

```text
Your app is live at: https://xyz123.ngrok.io
```
## 📸 Screenshots
### 📸 Screenshot 1
![Screenshot 1](reports_screenshots/app-ss-1.png)

### 📸 Screenshot 2
![Screenshot 2](reports_screenshots/app-ss-2.png)

---

## 🧠 Behind the Scenes

- **Vectorizer:** TF-IDF (Top 2500 features)  
- **Classifier:** XGBoost (optimized)  
- **Explanation:** SHAP (Waterfall plot for individual prediction)  

---

> Made with ❤️ by combining Python, NLP, XGBoost, SHAP, and Streamlit.
