import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .fake-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .real-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer with caching
@st.cache_resource
def load_models():
    """Load the trained model and vectorizer"""
    try:
        BASE_DIR = Path(__file__).parent
        model_path = BASE_DIR / "xgb_fake_job_model.pkl"
        vectorizer_path = BASE_DIR / "tfidf_vectorizer.pkl"
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)

# Load models
model, vectorizer, load_error = load_models()

# Header
st.markdown('<div class="main-header">🔍 Fake Job Post Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Job Posting Authenticity Analysis using XGBoost & NLP</div>', unsafe_allow_html=True)

# Check if models loaded successfully
if load_error:
    st.error(f"❌ Error loading models: {load_error}")
    st.stop()

# Sidebar - Information and Examples
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool uses **Machine Learning** and **Natural Language Processing** to detect potentially fraudulent job postings.
    
    **How it works:**
    1. Enter job posting text
    2. AI analyzes the content
    3. Get prediction with confidence score
    4. View SHAP explanation
    """)
    
    st.markdown("---")
    
    st.header("📊 Model Info")
    st.info("""
    - **Algorithm**: XGBoost Classifier
    - **Features**: TF-IDF (5000 features)
    - **Training**: SMOTE-balanced dataset
    - **Explainability**: SHAP values
    """)
    
    st.markdown("---")
    
    st.header("🚩 Red Flags")
    st.warning("""
    Watch out for:
    - Unrealistic salary promises
    - Vague job descriptions
    - No company information
    - Immediate hiring pressure
    - Requests for personal information
    - Work-from-home schemes
    """)
    
    st.markdown("---")
    
    st.header("💡 Example Texts")
    example_choice = st.selectbox(
        "Load example:",
        ["None", "Fake Example", "Real Example"]
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Job Posting")
    
    # Load example text if selected
    default_text = ""
    if example_choice == "Fake Example":
        default_text = "Earn $5000/week from home! No experience needed. Apply now and start earning today! Click here to get started immediately."
    elif example_choice == "Real Example":
        default_text = "Software Engineer position at our tech company. Required: 3+ years Python experience, bachelor's degree in Computer Science. Responsibilities include developing scalable web applications, collaborating with cross-functional teams, and maintaining code quality. Competitive salary and benefits package offered."
    
    user_input = st.text_area(
        "Paste the job posting text below:",
        height=300,
        placeholder="Enter job description, requirements, benefits, or any text from the job posting...",
        value=default_text,
        help="The more text you provide, the better the analysis"
    )
    
    # Character count
    char_count = len(user_input)
    st.caption(f"📊 Character count: {char_count}")
    
    if char_count > 0 and char_count < 50:
        st.markdown('<div class="warning-box">⚠️ Short text may result in less accurate predictions. Try to include more details.</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 🎯 Quick Stats")
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin:0;">Text Length</h3>
        <h1 style="margin:0.5rem 0;">{len(user_input.split())}</h1>
        <p style="margin:0;">words</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    if user_input.strip():
        words = user_input.lower().split()
        suspicious_words = ['free', 'earn', 'money', 'fast', 'easy', 'guaranteed', 'click', 'now', 'urgent']
        found_suspicious = [w for w in suspicious_words if w in words]
        
        if found_suspicious:
            st.markdown(f"""
            <div class="fake-card">
                <h4 style="margin:0;">⚠️ Suspicious Words</h4>
                <p style="margin:0.5rem 0;">{len(found_suspicious)} detected</p>
                <p style="margin:0; font-size:0.9rem;">{', '.join(found_suspicious[:5])}</p>
            </div>
            """, unsafe_allow_html=True)

# Analyze button
st.markdown("---")
analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])

with analyze_col2:
    analyze_button = st.button("🔍 Analyze Job Posting", type="primary", use_container_width=True)

# Analysis section
if analyze_button:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Vectorize input
        status_text.text("📊 Processing text...")
        progress_bar.progress(25)
        time.sleep(0.3)
        
        input_vector = vectorizer.transform([user_input])
        
        # Make prediction
        status_text.text("🤖 Analyzing with AI model...")
        progress_bar.progress(50)
        time.sleep(0.3)
        
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]
        
        progress_bar.progress(75)
        time.sleep(0.3)
        
        status_text.text("✨ Generating explanation...")
        progress_bar.progress(100)
        time.sleep(0.3)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Results in columns
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric(
                label="Prediction",
                value="FAKE ⚠️" if prediction == 1 else "REAL ✅",
                delta="Suspicious" if prediction == 1 else "Legitimate"
            )
        
        with res_col2:
            confidence = proba[prediction] * 100
            st.metric(
                label="Confidence",
                value=f"{confidence:.1f}%",
                delta=f"{'High' if confidence > 80 else 'Medium' if confidence > 60 else 'Low'} certainty"
            )
        
        with res_col3:
            risk_level = "High Risk" if (prediction == 1 and proba[1] > 0.8) else \
                        "Medium Risk" if (prediction == 1 and proba[1] > 0.6) else \
                        "Low Risk"
            st.metric(
                label="Risk Level",
                value=risk_level,
                delta="Be cautious" if prediction == 1 else "Looks good"
            )
        
        # Detailed result card
        st.markdown("---")
        if prediction == 1:
            st.markdown(f"""
            <div class="fake-card">
                <h2 style="margin:0;">🚨 FAKE Job Posting Detected</h2>
                <h3 style="margin:1rem 0;">Confidence: {proba[1]*100:.2f}%</h3>
                <p style="font-size:1.1rem; margin:0;">
                This job posting shows characteristics commonly found in fraudulent listings. 
                We recommend extreme caution and thorough verification before proceeding.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="warning-box">⚠️ <strong>Recommendation:</strong> Do not provide personal information, payment, or banking details. Research the company independently and verify through official channels.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="real-card">
                <h2 style="margin:0;">✅ REAL Job Posting Detected</h2>
                <h3 style="margin:1rem 0;">Confidence: {proba[0]*100:.2f}%</h3>
                <p style="font-size:1.1rem; margin:0;">
                This job posting appears to be legitimate based on our analysis. 
                However, always exercise due diligence when applying.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="info-box">💡 <strong>Tip:</strong> Even for legitimate-looking postings, verify the company, check reviews, and research the position before sharing personal information.</div>', unsafe_allow_html=True)
        
        # Probability breakdown
        st.markdown("---")
        st.markdown("### 📈 Probability Breakdown")
        
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.metric("Real Job Probability", f"{proba[0]*100:.2f}%")
            st.progress(float(proba[0]))
        
        with prob_col2:
            st.metric("Fake Job Probability", f"{proba[1]*100:.2f}%")
            st.progress(float(proba[1]))
        
        # SHAP Explanation
        st.markdown("---")
        st.markdown("### 🔬 AI Explainability (SHAP Analysis)")
        st.info("📊 This chart shows which words/features influenced the prediction. Red bars push toward 'Fake', blue bars push toward 'Real'.")
        
        with st.spinner("Generating SHAP explanation..."):
            try:
                # Convert sparse input to dense
                input_dense = input_vector.toarray()
                
                # Get feature names
                feature_names = vectorizer.get_feature_names_out()
                
                # Create minimal background data for TreeExplainer
                background_text = [
                    "This is a normal job posting with experience required.",
                    "Software engineer position at technology company.",
                    "Marketing manager role with competitive benefits."
                ]
                background = vectorizer.transform(background_text).toarray()
                
                # Create TreeExplainer
                explainer = shap.TreeExplainer(model, background)
                shap_values = explainer.shap_values(input_dense)
                
                # Get top contributing features
                shap_vals = shap_values[0]
                abs_vals = np.abs(shap_vals)
                top_indices = np.argsort(abs_vals)[-15:][::-1]
                
                # Create manual bar plot instead of waterfall (more stable)
                top_features = [feature_names[i] for i in top_indices]
                top_values = [shap_vals[i] for i in top_indices]
                colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in top_values]
                
                # Generate plot
                fig, ax = plt.subplots(figsize=(12, 8))
                y_pos = np.arange(len(top_features))
                ax.barh(y_pos, top_values, color=colors, alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_features, fontsize=10)
                ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
                ax.set_title('Feature Importance for This Prediction', fontsize=14, fontweight='bold', pad=20)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax.grid(axis='x', alpha=0.3)
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#ff6b6b', alpha=0.8, label='Pushes toward FAKE'),
                    Patch(facecolor='#4ecdc4', alpha=0.8, label='Pushes toward REAL')
                ]
                ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.markdown("""
                **How to read this chart:**
                - Each bar represents a word/feature from your text
                - **Red bars** (→ right) push the prediction toward "Fake"
                - **Blue bars** (← left) push the prediction toward "Real"
                - Longer bars = stronger influence on the prediction
                - The chart shows the top 15 most influential features
                """)
                
            except Exception as e:
                st.error(f"Could not generate SHAP explanation: {str(e)}")
                st.info("The prediction is still valid, but the detailed explanation is temporarily unavailable.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>Disclaimer:</strong> This tool provides predictions based on machine learning analysis. 
    Always verify job postings independently and use multiple sources before making decisions.</p>
    <p>🔒 Your data is not stored or shared. All analysis happens locally.</p>
    <p style="margin-top: 1rem;">Built with ❤️ using Streamlit, XGBoost, and SHAP</p>
</div>
""", unsafe_allow_html=True)