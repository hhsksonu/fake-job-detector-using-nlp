# Fake Job Post Detector – Streamlit App

A professional web application that uses **Natural Language Processing (NLP)** and **Machine Learning** to detect fake job postings in real-time. Built with Streamlit, powered by XGBoost and TF-IDF vectorization, with AI explainability through SHAP.

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fake-job-detector-using-nlp-op3wr9fxrao2tul767qoax.streamlit.app/)

---

## Features

**Real-Time Detection** – Paste any job posting and get instant predictions  
**Confidence Scores** – See prediction probability percentages  
**AI Explainability** – SHAP waterfall plots show which words influenced the decision  
**Professional UI** – Modern, gradient-styled interface with intuitive design  
**Smart Analysis** – Real-time suspicious word detection and text statistics  
**Helpful Insights** – Built-in examples and red flag indicators  
**Privacy First** – All processing happens locally, no data stored  

---

## Files in This Directory

```
app/
├── app.py                      # Main Streamlit application
├── README.md                   # This file
├── requirements.txt            # App-specific dependencies
├── xgb_fake_job_model.pkl     # Trained XGBoost classifier
└── tfidf_vectorizer.pkl       # TF-IDF vectorizer (5000 features)
```

---

## Quick Start

### Option 1: Use Live Deployment (Easiest)

Simply visit the live app:  
**[https://fake-job-detector-using-nlp-op3wr9fxrao2tul767qoax.streamlit.app/](https://fake-job-detector-using-nlp-op3wr9fxrao2tul767qoax.streamlit.app/)**

> **Note:** If the app appears blank on first load, refresh the page once.

---

### Option 2: Run Locally

#### Prerequisites
- Python 3.8 or higher
- pip package manager

#### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 2: Launch the App

```bash
streamlit run app.py
```

The app will automatically open in your browser at:
```
http://localhost:8501
```

---

## How to Use the App

### 1. Input Job Posting
- Paste the job description in the text area
- Or load example text from the sidebar

### 2. Analyze
- Click the "Analyze Job Posting" button
- Watch the progress bar as AI processes your text

### 3. View Results
- **Prediction:** REAL or FAKE
- **Confidence Score:** How certain the model is
- **Risk Level:** Overall assessment
- **Probability Breakdown:** Visual representation of both classes

### 4. Understand the Decision
- **SHAP Waterfall Plot:** Shows which words pushed toward fake/real
- **Feature Explanation:** Learn what influenced the prediction

---

## App Interface Overview

### Main Features

#### Smart Input Area
- Real-time character/word counter
- Warning for short texts
- Suspicious word detector
- Quick stats dashboard

#### Analysis Dashboard
- 3-metric overview (Prediction, Confidence, Risk)
- Color-coded result cards
- Probability breakdown with progress bars
- Detailed recommendations

#### SHAP Visualization
- Interactive waterfall plot
- Feature importance ranking
- Easy-to-understand explanations
- High-quality matplotlib charts

#### Informative Sidebar
- About section
- Model information
- Red flags to watch for
- Example text loader

---

## Screenshots

### Main Interface
![App Screenshot 1](https://github.com/hhsksonu/fake-job-detector-using-nlp/blob/main/reports_screenshots/app-ss-1.png)

### Analysis Results
![App Screenshot 2](https://github.com/hhsksonu/fake-job-detector-using-nlp/blob/main/reports_screenshots/app-ss-2.png)

### SHAP Analysis Explainability 
![App Screenshot 2](https://github.com/hhsksonu/fake-job-detector-using-nlp/blob/main/reports_screenshots/app-ss-3.png)

---

## Technical Details

### Model Architecture
- **Vectorizer:** TF-IDF (5000 features, 1-2 word ngrams)
- **Classifier:** XGBoost with SMOTE balancing
- **Performance:** ~96% accuracy on test set
- **Explainability:** SHAP (SHapley Additive exPlanations)

### Key Technologies
```python
streamlit==1.20.0+      # Web framework
xgboost==1.5.0+        # Gradient boosting
scikit-learn==1.0.0+   # ML utilities
shap==0.41.0+          # Model explainability
matplotlib==3.4.0+     # Visualization
```

### How It Works

1. **Text Input** – User pastes job posting
2. **Preprocessing** – Cleaned using same pipeline as training
3. **Vectorization** – Converted to TF-IDF features (5000 dimensions)
4. **Prediction** – XGBoost model outputs probability
5. **Explanation** – SHAP computes feature contributions
6. **Visualization** – Results displayed with interactive plots

---

## UI/UX Highlights

### Design Principles
- **Clean & Modern** – Gradient color schemes and professional typography
- **Responsive** – Works on desktop, tablet, and mobile
- **Intuitive** – Clear call-to-actions and logical flow
- **Informative** – Contextual help and explanations
- **Accessible** – High contrast and readable fonts

### Color Coding
- **Blue/Teal** – Real job predictions
- **Red/Pink** – Fake job predictions
- **Purple** – Neutral/informational elements
- **Yellow** – Warnings and tips

---

## Common Red Flags (Built-In Detection)

The app automatically highlights these suspicious words:
- Financial promises: "earn", "money", "fast", "easy"
- Urgency markers: "now", "urgent", "immediately", "limited"
- Unrealistic offers: "free", "guaranteed", "no experience"
- Call-to-actions: "click", "apply now", "register"

---

## Troubleshooting

### App Won't Start?
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Try running with verbose mode
streamlit run app.py --logger.level=debug
```

### SHAP Plots Not Showing?
```bash
# Install/upgrade matplotlib backend
pip install --upgrade matplotlib

# Clear Streamlit cache
streamlit cache clear
```

### Model Files Missing?
Ensure `xgb_fake_job_model.pkl` and `tfidf_vectorizer.pkl` are in the same directory as `app.py`. If missing, run the training notebook (`02_model_training.ipynb`) to regenerate them.

---

## Dependencies

### Core Requirements
```
streamlit>=1.20.0
joblib>=1.1.0
scikit-learn>=1.0.0
xgboost>=1.5.0
shap>=0.41.0
matplotlib>=3.4.0
numpy>=1.21.0
```

Full list in `requirements.txt`

---

## Deployment

### Deploy to Streamlit Cloud (Free)

1. Push your code to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Select `app/app.py` as the main file
5. Deploy!

### Deploy to Other Platforms

- **Heroku:** Add `setup.sh` and `Procfile`
- **AWS/GCP:** Use container deployment
- **Local Network:** Use `--server.address` flag

---

## Tips for Best Results

1. **Include Complete Text** – More text = better accuracy
2. **Look for Context** – Check company info and job details
3. **Cross-Reference** – Verify the prediction with manual checks
4. **Trust the Confidence** – Higher confidence = more reliable
5. **Examine SHAP Plot** – Understand why the model decided

---

## Privacy & Security

- No data is stored or logged
- All processing happens in-memory
- No external API calls (except model loading)
- Open-source code for transparency

---

## Contributing

Found a bug or have a suggestion? Feel free to:
- Open an issue on GitHub
- Submit a pull request
- Contact the developer directly

---

## Related Documentation

- **Main Project README:** `../README.md`
- **Training Notebooks:** `../notebooks/`
- **Results & Analysis:** `../results/`

---

## Developer

**Sonu Kumar**  
Machine Learning Engineer | NLP Enthusiast

---

## Contact & Support

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hhsksonu)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/hhsksonu)

---

## Support This Project

If you find this app useful:
- Give it a star on [GitHub](https://github.com/hhsksonu/fake-job-detector-using-nlp)
- Share it with job seekers who might benefit
- Report any issues or suggest improvements

---

<div align="center">
  <p><strong>Built with care using Streamlit, XGBoost, and SHAP</strong></p>
  <p>Helping job seekers stay safe from fraudulent postings</p>
  <p>© 2024 Sonu Kumar</p>
</div>
