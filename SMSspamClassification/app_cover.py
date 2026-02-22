import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SMS Spam Classification",
    page_icon="📱",
    layout="centered"
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'page' not in st.session_state:
    st.session_state.page = 'cover'

# ============================================================
# LOAD MODEL FUNCTION
# ============================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('spam_classifier_lr.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        return model, vectorizer, feature_cols
    except Exception as e:
        return None, None, None

# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================
def extract_features(message):
    message = str(message)
    features = {
        'contains_urgent': 1 if re.search(r'urgent|urg|asap', message, re.I) else 0,
        'contains_winner': 1 if re.search(r'won|winner|win|claim|prize|cash', message, re.I) else 0,
        'contains_free': 1 if re.search(r'free|complimentary', message, re.I) else 0,
        'contains_call': 1 if re.search(r'call|text|reply', message, re.I) else 0,
        'contains_money': 1 if re.search(r'\$|£|€|pound|dollar', message, re.I) else 0,
        'contains_number': 1 if re.search(r'\d', message) else 0,
        'exclamation_count': message.count('!'),
        'question_count': message.count('?'),
        'capital_ratio': sum(1 for c in message if c.isupper()) / (len(message) + 1)
    }
    return features

# ============================================================
# COVER PAGE
# ============================================================
def show_cover():
    st.markdown("<h1 style='text-align: center; font-size: 5rem;'>📱</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #333;'>SMS Spam Classification</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>Analyze SPAM & HAM messages with Machine Learning</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍")
        st.markdown("**Smart Detection**")
        st.markdown("98% accuracy")
    with col2:
        st.markdown("### ⚡")
        st.markdown("**Instant Results**")
        st.markdown("Real-time analysis")
    with col3:
        st.markdown("### 🛡️")
        st.markdown("**Safe & Secure**")
        st.markdown("Protect your inbox")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 GET STARTED", use_container_width=True, type="primary"):
            st.session_state.page = 'main'
            st.rerun()

# ============================================================
# MAIN APP
# ============================================================
def show_main():
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back"):
            st.session_state.page = 'cover'
            st.rerun()
    with col2:
        st.title("📱 SMS Spam Classifier")
    
    st.markdown("---")
    
    with st.spinner("Loading model..."):
        model, vectorizer, feature_cols = load_model()
    
    if model is None:
        st.error("❌ Could not load model. Please make sure model files exist:")
        st.code("Required files:\n- spam_classifier_lr.pkl\n- tfidf_vectorizer.pkl\n- feature_columns.pkl")
        return
    
    st.success("✅ Model ready!")
    
    st.subheader("📝 Enter your message:")
    message = st.text_area(
        "",
        height=150,
        placeholder="Type your SMS message here...",
        key="input_message"
    )
    
    if st.button("🔍 ANALYZE MESSAGE", type="primary", use_container_width=True):
        if not message or message.strip() == "":
            st.warning("Please enter a message to analyze.")
        else:
            with st.spinner("Analyzing..."):
                cleaned_msg = clean_text(message)
                features = extract_features(message)
                
                msg_tfidf = vectorizer.transform([cleaned_msg])
                features_array = np.array([[features[col] for col in feature_cols]])
                final_features = hstack([msg_tfidf, features_array])
                
                prediction = model.predict(final_features)[0]
                proba = model.predict_proba(final_features)[0]
                confidence = proba[prediction]
                
                st.subheader("📋 Result:")
                
                if prediction == 1:
                    st.error(f"🚨 **SPAM DETECTED!** (Confidence: {confidence:.2%})")
                    
                    st.subheader("⚠️ Spam Indicators:")
                    indicators = []
                    if features['contains_urgent']: indicators.append("• Urgent language")
                    if features['contains_winner']: indicators.append("• Prize/winner language")
                    if features['contains_free']: indicators.append("• 'Free' offers")
                    if features['contains_call']: indicators.append("• Call to action")
                    if features['contains_money']: indicators.append("• Money symbols")
                    if features['exclamation_count'] > 2: indicators.append(f"• {features['exclamation_count']} exclamation marks")
                    
                    if indicators:
                        for ind in indicators:
                            st.write(ind)
                    else:
                        st.write("No obvious indicators, but model classified as spam")
                else:
                    st.success(f"✅ **LEGITIMATE MESSAGE (HAM)** (Confidence: {confidence:.2%})")
    
    st.markdown("---")
    st.markdown("<div style='text-align: center'><p>Built with ❤️ using Machine Learning</p></div>", unsafe_allow_html=True)

# ============================================================
# MAIN APP ROUTER
# ============================================================
if st.session_state.page == 'cover':
    show_cover()
else:
    show_main()