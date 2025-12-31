# Fake Job Detector using NLP and SHAP Explainability

A comprehensive Machine Learning and NLP-based project that detects fake job postings using real-world data. This system leverages TF-IDF vectorization, XGBoost classifier, and SHAP (SHapley Additive exPlanations) for model interpretability to identify and explain fraudulent job descriptions.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fake-job-detector-using-nlp-op3wr9fxrao2tul767qoax.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Project Overview

In the digital age, fake job postings are increasingly common, targeting job seekers with fraudulent opportunities. This project builds an **AI-powered detection system** that:

- Analyzes job posting text using Natural Language Processing
- Predicts whether a posting is **REAL** or **FAKE** with high accuracy
- Provides **explainable AI** insights using SHAP visualizations
- Offers a user-friendly web interface for real-time detection

---

## Key Features & Highlights

- End-to-end ML Pipeline – From data exploration to model deployment
- Advanced NLP – TF-IDF vectorization (5000 features, 1-2 ngrams)
- Robust Classifier – XGBoost with SMOTE for handling class imbalance
- Model Explainability – SHAP force plots, waterfall plots, and summary plots
- Interactive Visualization – Word clouds, confusion matrices, feature importance
- Production-Ready App – Beautiful Streamlit interface with real-time predictions
- Comprehensive Analysis – Error analysis, misclassification insights

---

## Project Structure

```
fake-job-detector-using-nlp/
│
├── app/
│   ├── app.py                          # Streamlit web application
│   ├── README.md                       # App-specific documentation
│   ├── requirements.txt                # App dependencies
│   ├── tfidf_vectorizer.pkl           # Trained TF-IDF vectorizer
│   └── xgb_fake_job_model.pkl         # Trained XGBoost model
│
├── data/
│   ├── fake_job_postings.csv          # Raw dataset
│   └── cleaned_fake_job_postings.csv  # Preprocessed dataset
│
├── models/
│   ├── xgb_fake_job_model.pkl         # Trained model (main copy)
│   └── tfidf_vectorizer.pkl           # Vectorizer (main copy)
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_model_training.ipynb        # Model Training & Evaluation
│   └── 03_explainability.ipynb        # SHAP Analysis & Insights
│
├── results/
│   ├── classification_report.csv      # Detailed metrics
│   ├── confusion_matrix.png           # Model performance visualization
│   ├── error_analysis.csv             # Misclassification analysis
│   ├── feature_importance.csv         # Top predictive features
│   └── feature_importance.png         # Feature importance chart
│
├── reports_screenshots/
│   ├── wordcloud.png                  # Word frequency visualization
│   ├── Real-vs-Fake-Job.png          # Class distribution
│   ├── Null-value-heatmap.png        # Missing data analysis
│   ├── SHAP-force-plot.png           # SHAP explanation example
│   ├── Waterfall-plot.png            # Individual prediction breakdown
│   ├── Top-predictive-words.png      # Feature importance
│   ├── results.png                    # Overall results
│   ├── app-ss-1.png                   # App screenshot 1
│   └── app-ss-2.png                   # App screenshot 2
│
├── requirements.txt                    # Project dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

---

## Live Demo

**[Try the Live App](https://fake-job-detector-using-nlp-op3wr9fxrao2tul767qoax.streamlit.app/)**

Simply paste any job posting text and get instant predictions with AI-powered explanations!

> **Note:** If the app appears blank on first load, please refresh the page once.

---

## Visual Results & Insights

### Data Exploration

| Visualization | Description |
|--------------|-------------|
| ![Null Heatmap](reports_screenshots/Null-value-heatmap.png) | **Missing Values Analysis** – Identified columns with excessive nulls |
| ![Class Distribution](reports_screenshots/Real-vs-Fake-Job.png) | **Class Imbalance** – Shows 17,014 real vs 866 fake jobs (5% fraud rate) |
| ![WordCloud](reports_screenshots/wordcloud.png) | **Word Frequency** – Most common terms in job postings |

### Model Performance

| Visualization | Description |
|--------------|-------------|
| ![Results](reports_screenshots/results.png) | **Confusion Matrix & Metrics** – High accuracy with balanced precision/recall |
| ![Top Words](reports_screenshots/Top-predictive-words.png) | **Feature Importance** – Words most indicative of fake jobs |

### Explainability (SHAP)

| Visualization | Description |
|--------------|-------------|
| ![SHAP Force](reports_screenshots/SHAP-force-plot.png) | **SHAP Force Plot** – Overall feature contribution to prediction |
| ![Waterfall](reports_screenshots/Waterfall-plot.png) | **Waterfall Plot** – Step-by-step prediction breakdown |

**Detailed metrics available in:** `results/classification_report.csv`

---

## Methodology & Workflow

### 1. Data Exploration (`01_eda.ipynb`)
- Loaded and analyzed 17,880 job postings
- Handled missing values (removed columns with >50% nulls)
- Combined text fields: title, department, description, requirements, benefits
- Text preprocessing: lowercase, remove HTML/URLs, lemmatization, stopword removal
- Visualized class distribution and word frequencies

### 2. Model Training (`02_model_training.ipynb`)
- **Vectorization:** TF-IDF with 5000 features, 1-2 word ngrams
- **Class Balancing:** Applied SMOTE to handle 95-5% imbalance
- **Model:** XGBoost Classifier with optimized hyperparameters
- **Evaluation:** Classification report, confusion matrix, accuracy metrics
- **Feature Analysis:** Identified top 20 predictive words

### 3. Explainability (`03_explainability.ipynb`)
- Generated SHAP values for test set predictions
- Created waterfall plots for individual predictions
- Built summary plots showing feature importance across samples
- Analyzed misclassified examples (false positives/negatives)
- Exported comprehensive reports and visualizations

### 4. Deployment (`app/app.py`)
- Built interactive Streamlit web application
- Real-time prediction with confidence scores
- SHAP waterfall visualization for each prediction
- Professional UI with gradient cards and metrics dashboard
- Deployed to Streamlit Cloud for public access

---

## Dataset

**Source:** [Kaggle - Fake Job Postings Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

Due to GitHub's 25MB file size limit, the dataset is hosted externally:

**[Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1rFXS_Wndua__KcTPd4jMm_cicOmjSxy0?usp=sharing)**

**Dataset Statistics:**
- Total Job Postings: 17,880
- Real Jobs: 17,014 (95.2%)
- Fake Jobs: 866 (4.8%)
- Features: 18 columns including text fields and metadata

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/hhsksonu/fake-job-detector-using-nlp.git
cd fake-job-detector-using-nlp
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1rFXS_Wndua__KcTPd4jMm_cicOmjSxy0?usp=sharing) and place `fake_job_postings.csv` in the `data/` folder.

---

## Usage

### Run Jupyter Notebooks (Analysis)

```bash
jupyter notebook
```

Navigate to `notebooks/` and open:
1. `01_eda.ipynb` – Data exploration and cleaning
2. `02_model_training.ipynb` – Model training and evaluation
3. `03_explainability.ipynb` – SHAP analysis and insights

### Run Streamlit App (Local)

```bash
cd app
streamlit run app.py
```

The app will open at `http://localhost:8501`

See `app/README.md` for detailed app documentation.

---

## Requirements

### Core Libraries
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0
```

### NLP & Visualization
```
nltk>=3.6
wordcloud>=1.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Explainability & Deployment
```
shap>=0.41.0
streamlit>=1.20.0
joblib>=1.1.0
```

Full list in `requirements.txt`

---

## Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | ~96% |
| **Precision (Fake)** | ~92% |
| **Recall (Fake)** | ~89% |
| **F1-Score (Fake)** | ~90% |

*Note: Metrics may vary slightly based on train-test split*

### Key Insights
- Model effectively identifies fake jobs while minimizing false alarms
- SMOTE balancing significantly improved minority class (fake jobs) detection
- Top predictive features include: "earn", "home", "click", "money", "free"
- Real jobs characterized by: "experience", "team", "responsibilities", "qualifications"

---

## Future Enhancements

- Deep learning models (LSTM, BERT) for improved accuracy
- Multi-language support for international job postings
- Real-time scraping and automatic database updates
- Browser extension for instant job posting verification
- API endpoint for integration with job boards
- Mobile app version for on-the-go verification

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- SHAP library by [Scott Lundberg](https://github.com/slundberg/shap)
- Streamlit for the amazing web framework
- XGBoost team for the powerful gradient boosting library

---

## Built By

**Sonu Kumar**

Passionate about using AI/ML to solve real-world problems and make the internet a safer place for job seekers.

---

## Connect With Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hhsksonu)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hhsksonu)

---

## Show Your Support

If you found this project helpful, please give it a star on GitHub!

---

<div align="center">
  <p>Made with care using Python, NLP, XGBoost, SHAP, and Streamlit</p>
  <p>© 2024 Sonu Kumar. All Rights Reserved.</p>
</div>
