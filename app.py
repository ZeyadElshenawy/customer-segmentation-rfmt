import streamlit as st
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="RFMT Cluster Predictor",
    page_icon="🎯",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_models():
    """Load the pre-trained model and scaler"""
    try:
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return kmeans, scaler, True
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, False

# Prediction function
# Cluster names mapping
CLUSTER_NAMES = {
    0: 'Champions',        # High frequency, high monetary, low recency, high tenure
    1: 'At Risk',          # High recency (inactive), lower frequency
    2: 'Promising'         # Medium values across metrics
}

def predict_cluster(recency, frequency, monetary, tenure, kmeans, scaler):
    """Predict cluster for input RFMT values"""
    # Create input array
    input_data = np.array([[recency, frequency, monetary, tenure]])
    
    # Log transformation
    input_log = np.log(input_data + 1)
    
    # Standardization
    input_scaled = scaler.transform(input_log)
    
    # Prediction
    cluster = kmeans.predict(input_scaled)[0]
    
    return cluster

# Main app
st.markdown('<div class="main-header">🎯 RFMT Cluster Predictor</div>', unsafe_allow_html=True)

# Load models
kmeans, scaler, success = load_models()

if not success:
    st.error("⚠️ Could not load model files. Please ensure the following files are in the same directory:")
    st.write("- kmeans_model.pkl")
    st.write("- scaler.pkl")
    st.stop()

st.success("✅ Model loaded successfully!")

# Input section
st.markdown("### Enter Customer RFMT Metrics")

col1, col2 = st.columns(2)

with col1:
    recency = st.number_input(
        "🕐 Recency (days)",
        min_value=0,
        max_value=1000,
        value=30,
        help="Days since last purchase"
    )
    
    frequency = st.number_input(
        "🔄 Frequency (orders)",
        min_value=1,
        max_value=1000,
        value=5,
        help="Total number of orders"
    )

with col2:
    monetary = st.number_input(
        "💰 Monetary Value ($)",
        min_value=0.01,
        max_value=1000000.0,
        value=500.0,
        step=10.0,
        help="Total spending"
    )
    
    tenure = st.number_input(
        "📅 Tenure (days)",
        min_value=1,
        max_value=10000,
        value=180,
        help="Days as customer"
    )

# Predict button
if st.button("🔮 Predict Cluster", type="primary", use_container_width=True):
    
    # Make prediction
    cluster = predict_cluster(recency, frequency, monetary, tenure, kmeans, scaler)
    cluster_name = CLUSTER_NAMES.get(cluster, f'Cluster {cluster}')
    
    # Emoji mapping
    emoji_map = {
        'Champions': '👑',
        'At Risk': '⚠️',
        'Promising': '⭐'
    }
    emoji = emoji_map.get(cluster_name, '🎯')
    
    # Display result
    st.markdown(f"""
        <div class="result-card">
            <h1 style="margin:0; font-size: 4rem;">{emoji}</h1>
            <h2 style="margin: 1rem 0; font-size: 2.5rem;">{cluster_name}</h2>
            <p style="margin: 0.5rem 0; font-size: 1.2rem; opacity: 0.9;">Cluster {cluster}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show input summary
    st.markdown("### 📋 Input Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Recency", f"{recency} days")
    
    with col2:
        st.metric("Frequency", f"{frequency} orders")
    
    with col3:
        st.metric("Monetary", f"${monetary:.2f}")
    
    with col4:
        st.metric("Tenure", f"{tenure} days")

# Sidebar info
with st.sidebar:
    st.markdown("### 📖 About RFMT")
    st.info(
        "**Recency**: Days since last purchase\n\n"
        "**Frequency**: Number of purchases\n\n"
        "**Monetary**: Total spending amount\n\n"
        "**Tenure**: Days as a customer"
    )
    
    st.markdown("---")
    
    if kmeans is not None:
        st.markdown("### 🎯 Model Info")
        st.write(f"**Clusters**: {kmeans.n_clusters}")
        st.write(f"**Algorithm**: K-Means")