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
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'page' not in st.session_state:
    st.session_state.page = 'cover'
if 'history' not in st.session_state:
    st.session_state.history = []
if 'history_df' not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=['Time', 'Message', 'Prediction', 'Confidence', 'Spam_Probability'])

# ============================================================
# 3D COLORFUL BOXES CSS
# ============================================================
def add_3d_boxes_css():
    st.markdown("""
        <style>
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        }
        
        /* Title styles */
        .main-title {
            text-align: center;
            font-size: 5rem;
            text-shadow: 0 0 20px #ff6b6b, 0 0 40px #4ecdc4;
            animation: glowPulse 3s infinite;
        }
        
        @keyframes glowPulse {
            0%, 100% { text-shadow: 0 0 20px #ff6b6b, 0 0 40px #4ecdc4; }
            50% { text-shadow: 0 0 30px #4ecdc4, 0 0 60px #ff6b6b; }
        }
        
        /* Container for 3D boxes */
        .boxes-container {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 50px 0;
            perspective: 1000px;
        }
        
        /* 3D Box styles */
        .box-3d {
            flex: 1;
            padding: 30px 20px;
            border-radius: 20px;
            text-align: center;
            transform-style: preserve-3d;
            transition: all 0.5s ease;
            box-shadow: 
                0 20px 30px -10px rgba(0,0,0,0.5),
                0 0 0 2px rgba(255,255,255,0.1) inset;
            position: relative;
            cursor: pointer;
        }
        
        /* Individual box colors */
        .box-3d:nth-child(1) {
            background: linear-gradient(145deg, #ff6b6b, #ff8e8e);
            transform: rotateY(-5deg) rotateX(5deg) translateZ(20px);
        }
        
        .box-3d:nth-child(2) {
            background: linear-gradient(145deg, #4ecdc4, #45b7af);
            transform: rotateY(0deg) rotateX(5deg) translateZ(40px);
        }
        
        .box-3d:nth-child(3) {
            background: linear-gradient(145deg, #ffd93d, #ffb347);
            transform: rotateY(5deg) rotateX(5deg) translateZ(20px);
        }
        
        /* 3D Hover effects */
        .box-3d:hover {
            transform: translateY(-20px) rotateY(0deg) rotateX(10deg) translateZ(50px);
            box-shadow: 
                0 30px 40px -10px rgba(0,0,0,0.6),
                0 0 0 4px rgba(255,255,255,0.3) inset;
        }
        
        .box-3d:nth-child(1):hover {
            background: linear-gradient(145deg, #ff8e8e, #ff6b6b);
        }
        
        .box-3d:nth-child(2):hover {
            background: linear-gradient(145deg, #45b7af, #4ecdc4);
        }
        
        .box-3d:nth-child(3):hover {
            background: linear-gradient(145deg, #ffb347, #ffd93d);
        }
        
        /* Box content */
        .box-icon {
            font-size: 3rem;
            margin-bottom: 15px;
            filter: drop-shadow(0 10px 10px rgba(0,0,0,0.3));
            transform: translateZ(30px);
        }
        
        .box-title {
            color: white;
            font-size: 1.4rem;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            transform: translateZ(40px);
        }
        
        .box-desc {
            color: rgba(255,255,255,0.9);
            font-size: 1rem;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
            transform: translateZ(35px);
        }
        
        /* 3D Button */
        .stButton > button {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4) !important;
            color: white !important;
            border: none !important;
            font-size: 1.3rem !important;
            padding: 15px 40px !important;
            border-radius: 50px !important;
            font-weight: bold !important;
            box-shadow: 
                0 20px 30px -10px rgba(0,0,0,0.5),
                0 5px 0 #3a8b8b,
                0 0 0 2px rgba(255,255,255,0.3) inset !important;
            transform: translateY(-5px);
            transition: all 0.2s ease !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) scale(1.05) !important;
            box-shadow: 
                0 25px 35px -10px rgba(0,0,0,0.6),
                0 3px 0 #3a8b8b,
                0 0 0 4px rgba(255,255,255,0.3) inset !important;
        }
        
        .stButton > button:active {
            transform: translateY(5px) !important;
            box-shadow: 
                0 15px 25px -10px rgba(0,0,0,0.4),
                0 0px 0 #3a8b8b !important;
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #a0a0ff;
            font-size: 1.3rem;
            margin-bottom: 30px;
            text-shadow: 0 0 10px rgba(74,158,255,0.5);
        }
        
        /* Main title */
        h1, h2, h3 {
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
        }
        
        /* Divider */
        hr {
            background: linear-gradient(90deg, transparent, #ff6b6b, #4ecdc4, #ffd93d, transparent) !important;
            height: 3px !important;
            border: none !important;
        }
        
        /* Table styles */
        .dataframe {
            background: rgba(255,255,255,0.1) !important;
            color: white !important;
            border-radius: 10px !important;
        }
        
        /* Metric cards */
        .metric-card {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        /* History stats */
        .stat-box {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #ffd93d;
        }
        
        .stat-label {
            color: #a0a0a0;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

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
# COVER PAGE
# ============================================================
def show_cover():
    add_3d_boxes_css()
    
    # Title with glow effect
    st.markdown('<div class="main-title">📱</div>', unsafe_allow_html=True)
    st.title("SMS Spam Classifier")
    st.markdown('<div class="subtitle">Detect spam messages with Machine Learning</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 3D Boxes container
    st.markdown('<div class="boxes-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="box-3d">
                <div class="box-icon">🔍</div>
                <div class="box-title">Smart Detection</div>
                <div class="box-desc">98% accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="box-3d">
                <div class="box-icon">⚡</div>
                <div class="box-title">Instant Results</div>
                <div class="box-desc">Real-time analysis</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="box-3d">
                <div class="box-icon">📜</div>
                <div class="box-title">History</div>
                <div class="box-desc">Track all messages</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 GET STARTED", use_container_width=True):
            st.session_state.page = 'main'
            st.rerun()

# ============================================================
# MAIN APP WITH HISTORY
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
                st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-number">{total}</div>
                        <div class="stat-label">Total Messages</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-number" style="color: #ff6b6b;">{spam_count}</div>
                        <div class="stat-label">Spam Messages</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-number" style="color: #4ecdc4;">{ham_count}</div>
                        <div class="stat-label">Ham Messages</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-number" style="color: #ffd93d;">{spam_pct:.1f}%</div>
                        <div class="stat-label">Spam Rate</div>
                    </div>
                """, unsafe_allow_html=True)
            
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