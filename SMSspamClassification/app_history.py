import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from datetime import datetime
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
if 'history_df' not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=['Time', 'Message', 'Prediction', 'Confidence', 'Spam_Probability'])

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
# ADD TO HISTORY FUNCTION
# ============================================================
def add_to_history(message, prediction, confidence, spam_prob):
    """Add message to history"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Shorten message if too long
    short_msg = message[:50] + '...' if len(message) > 50 else message
    
    new_entry = pd.DataFrame([{
        'Time': timestamp,
        'Message': short_msg,
        'Prediction': 'SPAM' if prediction == 1 else 'HAM',
        'Confidence': f"{confidence:.2%}",
        'Spam_Probability': f"{spam_prob:.2%}"
    }])
    
    st.session_state.history_df = pd.concat([new_entry, st.session_state.history_df]).reset_index(drop=True)

# ============================================================
# CLEAR HISTORY FUNCTION
# ============================================================
def clear_history():
    """Clear all history"""
    st.session_state.history_df = pd.DataFrame(columns=['Time', 'Message', 'Prediction', 'Confidence', 'Spam_Probability'])

# ============================================================
# ORIGINAL COVER PAGE (YOUR DESIGN)
# ============================================================
def show_cover():
    st.markdown("<h1 style='text-align: center; font-size: 5rem;'>📱</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #333;'>SMS Spam Classification</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>Analyze spam & ham messages with Machine Learning</p>", unsafe_allow_html=True)
    
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
        st.markdown("### 📜")
        st.markdown("**History Tracking**")
        st.markdown("Save all tests")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 GET STARTED", use_container_width=True, type="primary"):
            st.session_state.page = 'main'
            st.rerun()

# ============================================================
# MAIN APP WITH HISTORY TABS
# ============================================================
def show_main():
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back"):
            st.session_state.page = 'cover'
            st.rerun()
    with col2:
        st.title("📱 SMS Spam Classifier")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Classifier", "📜 History", "📊 Statistics"])
    
    # ==================== TAB 1: CLASSIFIER ====================
    with tab1:
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
            height=120,
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
                    
                    # Add to history
                    add_to_history(message, prediction, confidence, proba[1])
                    
                    st.subheader("📋 Result:")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.error(f"🚨 **SPAM DETECTED!**")
                        else:
                            st.success(f"✅ **LEGITIMATE MESSAGE**")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Show probability bar
                    st.markdown("**Spam Probability:**")
                    st.progress(float(proba[1]))
                    st.markdown(f"{proba[1]:.2%}")
                    
                    if prediction == 1:
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
    
    # ==================== TAB 2: HISTORY ====================
    with tab2:
        st.subheader("📜 Message History")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                clear_history()
                st.rerun()
        
        st.markdown("---")
        
        if len(st.session_state.history_df) == 0:
            st.info("No history yet. Analyze some messages to see them here!")
        else:
            # Show history table
            st.dataframe(
                st.session_state.history_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = st.session_state.history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History as CSV",
                data=csv,
                file_name=f"spam_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # ==================== TAB 3: STATISTICS ====================
    with tab3:
        st.subheader("📊 History Statistics")
        
        if len(st.session_state.history_df) == 0:
            st.info("No data yet. Analyze some messages to see statistics!")
        else:
            df = st.session_state.history_df
            
            # Calculate stats
            total = len(df)
            spam_count = len(df[df['Prediction'] == 'SPAM'])
            ham_count = len(df[df['Prediction'] == 'HAM'])
            spam_pct = (spam_count/total)*100 if total > 0 else 0
            
            # Show metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Messages", total)
            with col2:
                st.metric("Spam", spam_count)
            with col3:
                st.metric("Ham", ham_count)
            with col4:
                st.metric("Spam Rate", f"{spam_pct:.1f}%")
            
            # Show chart
            st.subheader("Spam vs Ham Distribution")
            chart_data = pd.DataFrame({
                'Type': ['Spam', 'Ham'],
                'Count': [spam_count, ham_count]
            })
            st.bar_chart(chart_data.set_index('Type'))
    
    st.markdown("---")
    st.markdown("<div style='text-align: center'><p>Built with ❤️ using Machine Learning</p></div>", unsafe_allow_html=True)

# ============================================================
# MAIN APP ROUTER
# ============================================================
if st.session_state.page == 'cover':
    show_cover()
else:
    show_main()