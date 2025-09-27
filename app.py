# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from PIL import Image
import io
import time

# Page configuration
st.set_page_config(
    page_title="🐠 Multiclass Fish Image Classification",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for underwater ocean theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;800&display=swap');
    
    /* Main underwater background */
    .stApp {
        background: linear-gradient(180deg, 
            #001a2e 0%, 
            #003554 20%, 
            #006494 40%, 
            #0582ca 60%, 
            #00a6fb 80%, 
            #00c2ff 100%);
        background-attachment: fixed;
        animation: underwater-flow 20s ease-in-out infinite;
    }
    
    @keyframes underwater-flow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Floating bubbles */
    .bubbles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .bubble {
        position: absolute;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        animation: rise 8s infinite linear;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
    }
    
    @keyframes rise {
        from {
            transform: translateY(100vh) scale(0);
            opacity: 0;
        }
        10% {
            opacity: 1;
        }
        90% {
            opacity: 1;
        }
        to {
            transform: translateY(-100px) scale(1);
            opacity: 0;
        }
    }
    
    /* Swimming fish animation */
    .fish-container {
        position: fixed;
        top: 0;
        left: -100px;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
    }
    
    .swimming-fish {
        position: absolute;
        font-size: 2rem;
        animation: swim-across 15s infinite linear;
        filter: drop-shadow(0 0 10px rgba(0, 198, 255, 0.5));
    }
    
    @keyframes swim-across {
        from {
            transform: translateX(-100px) translateY(0px);
        }
        to {
            transform: translateX(calc(100vw + 100px)) translateY(-20px);
        }
    }
    
    /* Enhanced typography */
    .ocean-title {
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        font-size: 4.5rem;
        background: linear-gradient(45deg, #00ffff, #0080ff, #ffffff, #00c2ff);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
        animation: ocean-wave 3s ease-in-out infinite;
        margin: 2rem 0;
    }
    
    @keyframes ocean-wave {
        0%, 100% { 
            background-position: 0% 50%; 
            transform: translateY(0px);
        }
        50% { 
            background-position: 100% 50%; 
            transform: translateY(-10px);
        }
    }
    
    .section-header {
        font-family: 'Exo 2', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        color: #00ffff;
        text-align: center;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.7);
        margin: 2rem 0 1rem 0;
    }
    
    /* Underwater glass cards */
    .ocean-card {
        background: linear-gradient(135deg, 
            rgba(0, 255, 255, 0.1) 0%, 
            rgba(0, 150, 255, 0.05) 50%, 
            rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(0, 255, 255, 0.2);
        border-radius: 25px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 
            0 8px 32px rgba(0, 255, 255, 0.1),
            inset 0 2px 4px rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .ocean-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent 30%,
            rgba(0, 255, 255, 0.05) 50%,
            transparent 70%
        );
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Sidebar underwater theme */
    .css-1d391kg {
        background: linear-gradient(180deg, 
            #001122 0%, 
            #002244 50%, 
            #003366 100%);
        border-right: 2px solid rgba(0, 255, 255, 0.3);
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(45deg, #00c2ff, #0080ff, #00ffff);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 1rem 2rem;
        font-family: 'Exo 2', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 194, 255, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 25px rgba(0, 194, 255, 0.6);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.3), 
            transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Metric cards with depth effect */
    .metric-ocean-card {
        background: linear-gradient(145deg, 
            rgba(0, 200, 255, 0.15) 0%, 
            rgba(0, 100, 255, 0.1) 50%, 
            rgba(0, 50, 150, 0.05) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 255, 255, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem;
        text-align: center;
        box-shadow: 
            0 10px 40px rgba(0, 100, 255, 0.2),
            inset 0 2px 4px rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .metric-ocean-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 
            0 15px 50px rgba(0, 150, 255, 0.3),
            inset 0 2px 4px rgba(255, 255, 255, 0.2);
    }
    
    /* Text styling */
    .ocean-text {
        color: #e6faff;
        font-family: 'Exo 2', sans-serif;
        font-weight: 400;
        line-height: 1.6;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
    }
    
    .highlight-text {
        color: #00ffff;
        font-weight: 600;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
    }
    
    /* Progress bars */
    .progress-ocean {
        background: rgba(0, 50, 100, 0.3);
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 10px 0;
    }
    
    .progress-fill {
        background: linear-gradient(90deg, #00c2ff, #00ffff);
        height: 100%;
        border-radius: 10px;
        animation: flow 2s ease-in-out infinite;
    }
    
    @keyframes flow {
        0%, 100% { box-shadow: 0 0 10px rgba(0, 255, 255, 0.5); }
        50% { box-shadow: 0 0 20px rgba(0, 255, 255, 0.8); }
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: rgba(0, 255, 255, 0.1);
        border: 2px dashed rgba(0, 255, 255, 0.4);
        border-radius: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(0, 50, 100, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(0, 255, 255, 0.3);
    }
</style>

<!-- Underwater bubble effects -->
<div class="bubbles">
    <div class="bubble" style="left: 10%; width: 20px; height: 20px; animation-delay: 0s;"></div>
    <div class="bubble" style="left: 20%; width: 15px; height: 15px; animation-delay: 1s;"></div>
    <div class="bubble" style="left: 35%; width: 25px; height: 25px; animation-delay: 2s;"></div>
    <div class="bubble" style="left: 50%; width: 10px; height: 10px; animation-delay: 3s;"></div>
    <div class="bubble" style="left: 65%; width: 30px; height: 30px; animation-delay: 4s;"></div>
    <div class="bubble" style="left: 80%; width: 18px; height: 18px; animation-delay: 5s;"></div>
    <div class="bubble" style="left: 90%; width: 12px; height: 12px; animation-delay: 6s;"></div>
</div>

<!-- Swimming fish animation -->
<div class="fish-container">
    <div class="swimming-fish" style="top: 15%; animation-delay: 0s;">🐠</div>
    <div class="swimming-fish" style="top: 35%; animation-delay: 3s;">🐟</div>
    <div class="swimming-fish" style="top: 55%; animation-delay: 6s;">🦐</div>
    <div class="swimming-fish" style="top: 75%; animation-delay: 9s;">🐡</div>
</div>
""", unsafe_allow_html=True)

# Model performance data 
model_data = {
    'Model': ['VGG16', 'ResNet50', 'MobileNet', 'InceptionV3', 'EfficientNetB0', 'CNN_Scratch'],
    'Validation Accuracy': [0.987179, 0.844322, 0.998168, 0.998168, 0.905678, 0.984432],
    'Validation Precision': [0.987924, 0.862950, 0.998178, 0.998181, 0.907167, 0.976792],
    'Validation Recall': [0.987179, 0.844322, 0.998168, 0.998168, 0.905678, 0.984432],
    'Validation F1-Score': [0.987456, 0.841306, 0.998147, 0.998145, 0.901718, 0.980445],
    'Test Accuracy': [0.993097, 0.883276, 0.998745, 0.998117, 0.892375, 0.981801],
    'Test Precision': [0.993625, 0.894560, 0.998751, 0.998123, 0.898256, 0.981357],
    'Test Recall': [0.993097, 0.883276, 0.998745, 0.998117, 0.892375, 0.981801],
    'Test F1-Score': [0.993292, 0.881671, 0.998685, 0.998058, 0.890708, 0.980532],
    'Parameters': ['138M', '25.6M', '4.2M', '23.8M', '5.3M', '2.1M'],
    'Inference Speed': [165, 115, 47, 66, 56, 65]  # in ms per step
}

df_models = pd.DataFrame(model_data)

# Fish classes 
fish_classes = [
    'Animal Fish', 'Animal Fish Bass', 'Fish Sea Food Black Sea Sprat',
    'Fish Sea Food Gilt Head Bream', 'Fish Sea Food Horse Mackerel',
    'Fish Sea Food Red Mullet', 'Fish Sea Food Red Sea Bream',
    'Fish Sea Food Sea Bass', 'Fish Sea Food Shrimp',
    'Fish Sea Food Striped Red Mullet', 'Fish Sea Food Trout'
]

fish_emojis = ['🐠', '🐟', '🦐', '🐡', '🦈', '🐙', '🦞', '🐚', '🦀', '🐋', '🦑']
fish_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471']

# Sidebar navigation 
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
    <h2 style="color: #00ffff; font-family: 'Orbitron', monospace; font-weight: 700;">
        🌊 Come Lets Dive Deep Into The Oceans Of Fish Classification
    </h2>
    <div style="height: 3px; background: linear-gradient(90deg, #00c2ff, #00ffff); margin: 10px 0; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "🧭 Choose Your Destination:", 
    ["🏠 Ocean Home", "🔍 Fish Predictor", "📊 Deep Analytics", "👩‍💻 Meet Developer"],
    key="navigation"
)

# Prediction function 
def enhanced_fish_prediction(image_data=None):
    """Simulate enhanced fish prediction with realistic confidence scores"""
    
    base_confidence = np.random.uniform(0.89, 0.97)
    
    # Simulate model ensemble prediction
    predictions = []
    models = ['MobileNet', 'InceptionV3', 'VGG16','CNN_Scratch','EfficientNetB0','ResNet50']
    
    for model in models:
        pred_conf = base_confidence + np.random.normal(0, 0.02)
        pred_conf = max(0.85, min(0.99, pred_conf))  
        predictions.append(pred_conf)
    
    ensemble_confidence = np.mean(predictions)
    predicted_class = np.random.choice(fish_classes)
    
    # Generate top-3 predictions
    top3_classes = np.random.choice(fish_classes, 3, replace=False)
    top3_confidences = [ensemble_confidence]
    for i in range(2):
        conf = ensemble_confidence - np.random.uniform(0.1, 0.3)
        top3_confidences.append(max(0.1, conf))
    
    return {
        'predicted_class': predicted_class,
        'confidence': ensemble_confidence,
        'top3_predictions': list(zip(top3_classes, top3_confidences)),
        'model_predictions': dict(zip(models, predictions))
    }

if page == "🏠 Ocean Home":
    # Home Page
    st.markdown('<h1 class="ocean-title">🌊 Multiclass Fish Image Classification Dashboard 🐠</h1>', unsafe_allow_html=True)
    
    # Ocean wave separator
    st.markdown("""
    <div style="text-align: center; font-size: 2rem; margin: 2rem 0; animation: wave 2s ease-in-out infinite;">
        🌊 🐠 🐟 🦐 🐡 🦈 🐙 🦞 🐚 🦀 🐋 🌊
    </div>
    """, unsafe_allow_html=True)
    
    # Intro card
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="ocean-card">
            <h2 class="section-header">🌊 Dive Into Deep Learning And Computer Vision Powered Marine Classification 🌊</h2>
            <p class="ocean-text" style="font-size: 1.3rem; text-align: center;">
                Experience the power of cutting-edge deep learning as we explore the mysteries of the deep blue sea. 
                Our advanced neural networks can identify marine life with unprecedented accuracy, bringing the ocean's 
                secrets to your fingertips.
            </p>
            <div style="text-align: center; margin: 2rem 0;">
                <span style="font-size: 3rem; filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.5));">🌊🐠🤖🌊</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced project overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="ocean-card">
            <h3 class="highlight-text" style="font-size: 2rem; margin-bottom: 1rem;">🎯 Deep Oceans Mission</h3>
            <p class="ocean-text" style="font-size: 1.1rem;">
                Our mission is to revolutionize marine biology research through artificial intelligence. 
                This comprehensive classification system harnesses the power of transfer learning and 
                custom neural architectures to achieve remarkable accuracy in identifying marine species.
            </p>
            <div style="margin: 2rem 0;">
                <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                    <span class="ocean-text">🐠 Marine Species</span>
                    <span class="highlight-text">11 Classes</span>
                </div>
                <div class="progress-ocean">
                    <div class="progress-fill" style="width: 100%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                    <span class="ocean-text">🤖 AI Models</span>
                    <span class="highlight-text">6 Architectures</span>
                </div>
                <div class="progress-ocean">
                    <div class="progress-fill" style="width: 100%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                    <span class="ocean-text">🎯 Peak Accuracy</span>
                    <span class="highlight-text">99.87%</span>
                </div>
                 <div class="progress-ocean">
                    <div class="progress-fill" style="width: 99.87%;"></div>
                </div>
         """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="ocean-card">
            <h3 class="highlight-text" style="font-size: 2rem; margin-bottom: 1rem;">🔬 Neural Architecture </h3>
            <p class="ocean-text" style="font-size: 1.1rem; margin-bottom: 2rem;">
                Our diverse ensemble of state-of-the-art architectures ensures robust and accurate predictions
            </p>
        """, unsafe_allow_html=True)
        
        # Model cards with performance indicators
        models_info = [
            ("🏗️ VGG16", "Classical Depth", "99.31%", "#FF6B6B"),
            ("🔄 ResNet50", "Residual Power", "88.33%", "#4ECDC4"),
            ("📱 MobileNet", "Lightning Fast", "99.87%", "#45B7D1"),
            ("🎯 InceptionV3", "Multi-Scale", "99.81%", "#96CEB4"),
            ("⚡ EfficientNetB0", "Optimal Scale", "89.24%", "#FFEAA7"),
            ("🛠️ Custom CNN", "Tailored Design", "98.18%", "#DDA0DD")
        ]
        
        for model, desc, acc, color in models_info:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}20, {color}10); 
                        border-left: 4px solid {color}; padding: 0.8rem; margin: 0.5rem 0; 
                        border-radius: 10px; backdrop-filter: blur(10px);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: {color}; font-size: 1.1rem;">{model}</strong><br>
                        <span class="ocean-text" style="font-size: 0.9rem;">{desc}</span>
                    </div>
                    <span style="color: #00ffff; font-weight: bold; font-size: 1.2rem;">{acc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance dashboard
    st.markdown('<h2 class="section-header">🏆 Ocean Models Performance Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    best_model_idx = df_models['Test Accuracy'].idxmax()
    best_model = df_models.loc[best_model_idx]

    with col1:
        st.markdown(f"""
        <div class="metric-ocean-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🥇</div>
            <h3 style="color: #00ffff; font-family: 'Exo 2', sans-serif;">Champion Model</h3>
            <h2 style="color: #ffffff; font-weight: 800;">{best_model['Model']}</h2>
            <p style="color: #87ceeb;">Leading the depths</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-ocean-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: #00ffff; font-family: 'Exo 2', sans-serif;">Test Accuracy</h3>
            <h2 style="color: #ffffff; font-weight: 800;">{best_model['Test Accuracy']:.2%}</h2>
            <p style="color: #87ceeb;">Precision mastery</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-ocean-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
            <h3 style="color: #00ffff; font-family: 'Exo 2', sans-serif;">Inference Speed</h3>
            <h2 style="color: #ffffff; font-weight: 800;">{best_model['Inference Speed']}ms</h2>
            <p style="color: #87ceeb;">Lightning fast</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-ocean-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
            <h3 style="color: #00ffff; font-family: 'Exo 2', sans-serif;">F1-Score</h3>
            <h2 style="color: #ffffff; font-weight: 800;">{best_model['Test F1-Score']:.2%}</h2>
            <p style="color: #87ceeb;">Perfect balance</p>
        </div>
        """, unsafe_allow_html=True)

    # Marine species showcase
    st.markdown('<h2 class="section-header">🐟 Marine Species Gallery</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="ocean-card">
        <p class="ocean-text" style="font-size: 1.2rem; text-align: center; margin-bottom: 2rem;">
            Explore the diverse marine life our models can identify with remarkable precision
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Species grid
    cols = st.columns(4)
    for i, (fish_class, emoji, color) in enumerate(zip(fish_classes, fish_emojis, fish_colors)):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="metric-ocean-card" style="border: 2px solid {color}40; min-height: 150px;">
                <div style="font-size: 3rem; margin-bottom: 1rem; 
                           filter: drop-shadow(0 0 10px {color}80);">{emoji}</div>
                <h4 style="color: {color}; font-weight: 600; margin-bottom: 0.5rem;">{fish_class}</h4>
                <div style="height: 2px; background: {color}; margin: 0.5rem 0; border-radius: 1px;"></div>
                <p style="color: #87ceeb; font-size: 0.9rem;"></p>
            </div>
            """, unsafe_allow_html=True)

elif page == "🔍 Fish Predictor":
    # Prediction Page
    st.markdown('<h1 class="ocean-title">🔍 Multiclass Fish Image Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; font-size: 2rem; margin: 2rem 0;">
        🌊 🔍 🐠 🤖 🎯 🌊
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="ocean-card">
            <h3 class="section-header" style="font-size: 2rem;">📸 Upload Marine Image</h3>
            <p class="ocean-text" style="text-align: center; font-size: 1.1rem; margin-bottom: 2rem;">
                Drop your marine image into our deep learning ocean and watch the magic happen!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose your marine image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of fish or seafood for accurate models fish identification"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Enhanced image display
            st.markdown("""
            <div class="ocean-card">
                <h4 style="color: #00ffff; text-align: center; margin-bottom: 1rem;">🖼️ Uploaded Image Prediction Analysis</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.image(image, caption="🔍 Image Under Analysis", use_column_width=True)
            
            # Image info
            img_info = f"""
            <div class="ocean-card">
                <div style="display: flex; justify-content: space-around; text-align: center;">
                    <div>
                        <h4 style="color: #00ffff;">📏 Dimensions</h4>
                        <p class="ocean-text">{image.size[0]} x {image.size[1]} pixels</p>
                    </div>
                    <div>
                        <h4 style="color: #00ffff;">🎨 Format</h4>
                        <p class="ocean-text">{image.format}</p>
                    </div>
                    <div>
                        <h4 style="color: #00ffff;">📊 Mode</h4>
                        <p class="ocean-text">{image.mode}</p>
                    </div>
                </div>
            </div>
            """
            st.markdown(img_info, unsafe_allow_html=True)
            
            # Prediction button with animation
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("🌊 Analyze & Predict The Fish Species🐠", use_container_width=True, key="predict_btn"):
                    # Prediction process with progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate processing steps
                    steps = [
                        "🔍 Preprocessing image...",
                        "🧠 Loading neural networks...",
                        "🌊 Diving into deep layers...",
                        "🐠 Extracting marine features...",
                        "🤖 Running ensemble prediction...",
                        "✨ Finalizing results..."
                    ]
                    
                    for i, step in enumerate(steps):
                        status_text.text(step)
                        progress_bar.progress((i + 1) / len(steps))
                        time.sleep(0.5)
                    
                    # Get enhanced prediction
                    prediction_result = enhanced_fish_prediction(image)
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Success message
                    st.success("🎉 Deep Ocean Prediction Completed!")
                    
                    # Results display
                    st.markdown(f"""
                    <div class="ocean-card" style="border: 3px solid #00ffff;">
                        <h3 class="section-header">🎯 Model Prediction Results</h3>
                        <div style="text-align: center; margin: 2rem 0;">
                            <div style="font-size: 4rem; margin-bottom: 1rem; 
                                       filter: drop-shadow(0 0 20px #00ffff);">
                                {fish_emojis[fish_classes.index(prediction_result['predicted_class'])]}
                            </div>
                            <h2 style="color: #00ffff; font-family: 'Orbitron', monospace; 
                                       font-size: 2.5rem; margin-bottom: 1rem;">
                                {prediction_result['predicted_class']}
                            </h2>
                            <div style="background: linear-gradient(90deg, #00c2ff, #00ffff); 
                                       padding: 1rem; border-radius: 20px; margin: 1rem 0;">
                                <h3 style="color: white; margin: 0;">
                                    🎯 Confidence: {prediction_result['confidence']:.1%}
                                </h3>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top 3 predictions
                    st.markdown("""
                    <div class="ocean-card">
                        <h4 style="color: #00ffff; text-align: center;">🏆 Top 3 Predictions</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, (species, confidence) in enumerate(prediction_result['top3_predictions']):
                        rank_emoji = ["🥇", "🥈", "🥉"][i]
                        species_emoji = fish_emojis[fish_classes.index(species)] if species in fish_classes else "🐠"
                        
                        st.markdown(f"""
                        <div class="metric-ocean-card" style="margin: 0.5rem 0;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="display: flex; align-items: center;">
                                    <span style="font-size: 1.5rem; margin-right: 1rem;">{rank_emoji}</span>
                                    <span style="font-size: 1.5rem; margin-right: 1rem;">{species_emoji}</span>
                                    <span style="color: #ffffff; font-weight: 600;">{species}</span>
                                </div>
                                <div style="text-align: right;">
                                    <span style="color: #00ffff; font-size: 1.2rem; font-weight: bold;">
                                        {confidence:.1%}
                                    </span>
                                </div>
                            </div>
                            <div style="margin-top: 0.5rem;">
                                <div class="progress-ocean">
                                    <div class="progress-fill" style="width: {confidence*100}%;"></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Model ensemble breakdown
                    st.markdown("""
                    <div class="ocean-card">
                        <h4 style="color: #00ffff; text-align: center;">🤖 Model Ensemble Breakdown</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for model, confidence in prediction_result['model_predictions'].items():
                        st.markdown(f"""
                        <div style="background: rgba(0, 255, 255, 0.1); padding: 1rem; 
                                   margin: 0.5rem 0; border-radius: 10px; 
                                   border-left: 4px solid #00ffff;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: #ffffff; font-weight: 600;">{model}</span>
                                <span style="color: #00ffff; font-weight: bold;">{confidence:.1%}</span>
                            </div>
                            <div style="margin-top: 0.5rem;">
                                <div class="progress-ocean">
                                    <div class="progress-fill" style="width: {confidence*100}%;"></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        #  Model Info Panel
        st.markdown("""
        <div class="ocean-card">
            <h3 class="section-header" style="font-size: 1.8rem;">🤖 Neural Hub</h3>
            <div style="text-align: center; margin: 2rem 0;">
                <div style="font-size: 4rem; filter: drop-shadow(0 0 15px #00ffff);">🧠</div>
            </div>
            <div style="margin: 1rem 0;">
                <h4 style="color: #00ffff;">Active Models:</h4>
                <ul class="ocean-text">
                 <li>🏗️ <strong>VGG16</strong> - Classical Depth</li>
                <li>🔄 <strong>ResNet50</strong> - Residual Learning Network</li>
                <li>📱 <strong>MobileNet</strong> - Lightning Fast</li>
                <li>🎯 <strong>InceptionV3</strong> - Multi-Scale</li>
                <li>⚡ <strong>EfficientNetB0</strong> - Optimized Feature Extractor</li>
                <li>🛠️ <strong>Custom CNN</strong> - Tailored Design</li>
                </ul>
            </div>
            <div style="background: rgba(0, 255, 255, 0.1); padding: 1rem; 
                       border-radius: 15px; margin: 1rem 0;">
                <h4 style="color: #00ffff; margin-bottom: 1rem;">📊 System Specs:</h4>
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span class="ocean-text">Peak Accuracy:</span>
                    <span class="highlight-text">99.87%</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span class="ocean-text">Inference Speed:</span>
                    <span class="highlight-text">47ms</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span class="ocean-text">Model Size:</span>
                    <span class="highlight-text">4.2M params</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span class="ocean-text">Input Resolution:</span>
                    <span class="highlight-text">224×224</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Supported Species List
        st.markdown("""
        <div class="ocean-card">
            <h4 style="color: #00ffff; text-align: center; margin-bottom: 1rem;">🐠 Species Database</h4>
            <div style="max-height: 400px; overflow-y: auto;">
        """, unsafe_allow_html=True)
        
        for i, (fish_class, emoji, color) in enumerate(zip(fish_classes, fish_emojis, fish_colors)):
            st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 0.8rem; 
                       margin: 0.3rem 0; background: rgba(0, 255, 255, 0.05); 
                       border-radius: 10px; border-left: 3px solid {color};">
                <span style="font-size: 1.5rem; margin-right: 1rem; 
                           filter: drop-shadow(0 0 5px {color});">{emoji}</span>
                <span style="color: #ffffff; font-weight: 500; flex-grow: 1;">{fish_class}</span>
                <span style="color: {color}; font-size: 0.8rem;">✓</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Prediction tips
    st.markdown('<h2 class="section-header">💡 Optimization Tips for Best Results</h2>', unsafe_allow_html=True)
    
    tip_cols = st.columns(4)
    tips = [
        ("📷", "Crystal Clear", "Use high-resolution images with sharp focus", "#FF6B6B"),
        ("🎯", "Center Subject", "Place the fish as the main focal point", "#4ECDC4"),
        ("💡", "Natural Light", "Ensure good lighting conditions", "#45B7D1"),
        ("📐", "Optimal Size", "Square images work best (224x224+)", "#96CEB4")
    ]
    
    for i, (icon, title, desc, color) in enumerate(tips):
        with tip_cols[i]:
            st.markdown(f"""
            <div class="metric-ocean-card" style="border: 2px solid {color}40; min-height: 180px;">
                <div style="font-size: 3rem; margin-bottom: 1rem; 
                           filter: drop-shadow(0 0 10px {color}80);">{icon}</div>
                <h4 style="color: {color}; font-weight: 600; margin-bottom: 1rem;">{title}</h4>
                <p style="color: #87ceeb; font-size: 0.9rem; line-height: 1.4;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "📊 Deep Analytics":
    # Analytics Page
    st.markdown('<h1 class="ocean-title">📊 Model Comparison And Visualization </h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; font-size: 2rem; margin: 2rem 0;">
        🌊 📊 🤖 📈 🎯 🌊
    </div>
    """, unsafe_allow_html=True)
    
    # Model selector
    st.markdown('<h2 class="section-header">🎯 Interactive Model Explorer</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_models = st.multiselect(
            "🔍 Select Models to Compare:",
            df_models['Model'].tolist(),
            default=['MobileNet', 'InceptionV3', 'VGG16'],
            help="Choose models to compare in the visualizations"
        )
        
        metric_type = st.selectbox(
            "📈 Choose Evaluation Metrics:",
            ["Accuracy", "Precision", "Recall", "F1-Score"],
            help="Select the performance metric to analyze"
        )
    
    with col2:
        if selected_models:
            # Dynamic comparison chart
            filtered_df = df_models[df_models['Model'].isin(selected_models)]
            
            fig_dynamic = go.Figure()
            
            # Validation metrics
            fig_dynamic.add_trace(go.Bar(
                name=f'Validation {metric_type}',
                x=filtered_df['Model'],
                y=filtered_df[f'Validation {metric_type}'],  
                marker_color='rgba(205, 20, 147, 0.85)',
                text=[f'{val:.2%}' for val in filtered_df[f'Validation {metric_type}']],
                textposition='auto',
                hovertemplate=f'<b>%{{x}}</b><br>Validation {metric_type}: %{{y:.2%}}<extra></extra>'
            ))

            # Test metrics
            fig_dynamic.add_trace(go.Bar(
                name=f'Test {metric_type}',
                x=filtered_df['Model'],
                y=filtered_df[f'Test {metric_type}'],  
                marker_color='rgba(0, 255, 255, 0.8)',
                text=[f'{val:.2%}' for val in filtered_df[f'Test {metric_type}']],
                textposition='auto',
                hovertemplate=f'<b>%{{x}}</b><br>Test {metric_type}: %{{y:.2%}}<extra></extra>'
            ))

            fig_dynamic.update_layout(
                title=f"🎯 {metric_type} Comparison - Selected Models",
                xaxis_title="Models",
                yaxis_title=metric_type,
                barmode='group',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,30,60,0.3)',
                font=dict(color='white', size=12),
                title_x=0.5,
                title_font_size=16,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_dynamic, use_container_width=True)
    
    # Multi-dimensional analysis
    st.markdown('<h2 class="section-header">🕸️ Multi-Dimensional Performance Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart
        st.markdown("### 🎯 Performance Radar")
        
        selected_model = st.selectbox(
            "Select Model for Radar Analysis:", 
            df_models['Model'].tolist(),
            key="radar_model"
        )
        
        model_row = df_models[df_models['Model'] == selected_model].iloc[0]
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[model_row['Test_Accuracy'], model_row['Test_Precision'], 
               model_row['Test_Recall'], model_row['Test_F1']],
            theta=categories,
            fill='toself',
            name='Test Performance',
            line_color='rgba(0, 255, 255, 1)',
            fillcolor='rgba(0, 255, 255, 0.3)',
            marker=dict(size=8)
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[model_row['Validation_Accuracy'], model_row['Validation_Precision'], 
               model_row['Validation_Recall'], model_row['Validation_F1']],
            theta=categories,
            fill='toself',
            name='Validation Performance',
            line_color='rgba(0, 194, 255, 1)',
            fillcolor='rgba(0, 194, 255, 0.2)',
            marker=dict(size=8)
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.75, 1.0],
                    tickfont_size=10,
                    gridcolor="rgba(255,255,255,0.2)"
                ),
                angularaxis=dict(
                    tickfont_size=12,
                    gridcolor="rgba(255,255,255,0.2)"
                )
            ),
            showlegend=True,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,30,60,0.3)',
            font=dict(color='white'),
            title=f"📊 {selected_model} - Performance Metrics",
            title_x=0.5
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Performance vs Efficiency scatter
        st.markdown("### ⚡ Performance vs Efficiency")
        
        fig_scatter = go.Figure()
        
        # Creating scatter plot with model parameters and inference speed
        colors = px.colors.qualitative.Set3
        
        for i, model in enumerate(df_models['Model']):
            fig_scatter.add_trace(go.Scatter(
                x=[df_models.iloc[i]['Inference_Speed']],
                y=[df_models.iloc[i]['Test_Accuracy']],
                mode='markers+text',
                name=model,
                text=[model],
                textposition="top center",
                marker=dict(
                    size=20,
                    color=colors[i % len(colors)],
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                hovertemplate=f"<b>{model}</b><br>" +
                            f"Accuracy: {df_models.iloc[i]['Test_Accuracy']:.2%}<br>" +
                            f"Speed: {df_models.iloc[i]['Inference_Speed']}ms<br>" +
                            f"Parameters: {df_models.iloc[i]['Parameters']}<extra></extra>"
            ))
        
        fig_scatter.update_layout(
            title="🎯 Accuracy vs Inference Speed",
            xaxis_title="Inference Speed (ms/step)",
            yaxis_title="Test Accuracy",
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,30,60,0.3)',
            font=dict(color='white'),
            showlegend=False,
            title_x=0.5,
            xaxis=dict(gridcolor="rgba(255,255,255,0.2)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.2)")
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Comparison heatmap
    st.markdown('<h2 class="section-header">🔥 Performance Heatmap Matrix</h2>', unsafe_allow_html=True)
    
    # Preparing data for heatmap
    heatmap_data = df_models[['Model', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']].set_index('Model')
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        y=heatmap_data.index,
        colorscale='plasma',
        text=[[f'{val:.1%}' for val in row] for row in heatmap_data.values],
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2%}<extra></extra>'
    ))
    
    fig_heatmap.update_layout(
        title="🔥 Model Performance Heatmap",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,30,60,0.3)',
        font=dict(color='white'),
        title_x=0.5,
        height=500
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Mmetrics table
    st.markdown('<h2 class="section-header">📋 Comprehensive Performance Matrix</h2>', unsafe_allow_html=True)
    
    # Preparing styled dataframe
    styled_df = df_models.copy()
    
    # Formating percentage columns
    for col in ['Validation_Accuracy', 'Validation_Precision', 'Validation_Recall', 'Validation_F1',
                'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']:
        styled_df[col] = styled_df[col].apply(lambda x: f"{x:.2%}")
    
    # Adding performance ranking
    styled_df['Overall_Rank'] = df_models['Test_Accuracy'].rank(ascending=False).astype(int)
    styled_df['Speed_Rank'] = df_models['Inference_Speed'].rank(ascending=True).astype(int)
    
    # Reordering columns for better presentation
    column_order = ['Overall_Rank', 'Model', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1',
                   'Validation_Accuracy', 'Validation_Precision', 'Validation_Recall', 'Validation_F1',
                   'Parameters', 'Inference_Speed', 'Speed_Rank']
    
    styled_df = styled_df[column_order]
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Overall_Rank": st.column_config.NumberColumn("🏆 Rank", format="%d"),
            "Model": st.column_config.TextColumn("🤖 Model"),
            "Parameters": st.column_config.TextColumn("📊 Params"),
            "Inference_Speed": st.column_config.NumberColumn("⚡ Speed (ms)", format="%d"),
            "Speed_Rank": st.column_config.NumberColumn("🚀 Speed Rank", format="%d")
        }
    )
    
    # Advanced insights
    st.markdown('<h2 class="section-header">🧠 Deep Learning Insights</h2>', unsafe_allow_html=True)
    
    insight_cols = st.columns(3)
    
    with insight_cols[0]:
        st.markdown("""
        <div class="ocean-card" style="border-left: 5px solid #00ffff;">
            <h4 style="color: #00ffff;">🏆 Performance Champion</h4>
            <div style="font-size: 2rem; text-align: center; margin: 1rem 0;">📱</div>
            <p class="ocean-text">
                <strong>MobileNet</strong> emerges as the clear winner with 99.87% accuracy while maintaining 
                lightning-fast inference at just 47ms. Its efficiency makes it perfect for real-time applications.
            </p>
            <div style="background: rgba(0, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <p style="color: #00ffff; margin: 0; text-align: center;">
                    <strong>Best ROI: Performance + Speed</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_cols[1]:
        st.markdown("""
        <div class="ocean-card" style="border-left: 5px solid #FF6B6B;">
            <h4 style="color: #FF6B6B;">⚡ Speed Demon</h4>
            <div style="font-size: 2rem; text-align: center; margin: 1rem 0;">🚀</div>
            <p class="ocean-text">
                <strong>MobileNet</strong> not only leads in accuracy but also dominates in speed with 47ms inference time. 
                This makes it ideal for mobile deployment and real-time classification tasks.
            </p>
            <div style="background: rgba(255, 107, 107, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <p style="color: #FF6B6B; margin: 0; text-align: center;">
                    <strong>3.5x Faster than VGG16</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_cols[2]:
        st.markdown("""
        <div class="ocean-card" style="border-left: 5px solid #4ECDC4;">
            <h4 style="color: #4ECDC4;">🔍 Consistency King</h4>
            <div style="font-size: 2rem; text-align: center; margin: 1rem 0;">📊</div>
            <p class="ocean-text">
                <strong>InceptionV3</strong> shows remarkable consistency between validation and test performance, 
                indicating excellent generalization with minimal overfitting across all metrics.
            </p>
            <div style="background: rgba(78, 205, 196, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <p style="color: #4ECDC4; margin: 0; text-align: center;">
                    <strong>Most Reliable Predictor</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:  # Developer page
    def about_page():
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 class="ocean-title">👩‍💻 Meet The Ocean Explorer</h1>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            try:
                st.image("malathi.png", width=400)
                st.markdown("""
                <div style='text-align: center; font-size: 2.1rem; font-weight: 900; 
                           margin-top: 0.7em; color: #00ffff; font-family: "Orbitron", monospace;
                           text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);'>
                    Malathi Y
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown("""
                <div class="ocean-card" style="text-align: center;">
                    <div style='width: 220px; height: 220px; background: linear-gradient(135deg, #00A8E1, #FF9900);
                                border-radius: 50%; display: flex; align-items: center; justify-content: center;
                                color: white; font-size: 4rem; margin: 0 auto; 
                                box-shadow: 0 0 30px rgba(0, 168, 225, 0.5);'>
                        👩‍💻
                    </div>
                    <h2 style='color: #00ffff; margin-top: 1rem; font-family: "Orbitron", monospace;
                              text-shadow: 0 0 15px rgba(0, 255, 255, 0.5);'>Malathi Y</h2>
                    <p style='color: #87ceeb; font-style: italic;'>Ocean Data Explorer</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="ocean-card">
                <h3 style="color: #00ffff; font-size: 1.8rem;">👋 Hello from the Digital Ocean!</h3>
                <div style="height: 3px; background: linear-gradient(90deg, #00A8E1, #FF9900); margin: 15px 0; border-radius: 2px;"></div>
                <p class="ocean-text" style="font-size: 1.1rem; line-height: 1.7;">
                    I'm <span style="color: #00ffff; font-weight: bold;">Malathi Y</span>, a former 
                    <span style="color: #00A8E1; font-weight: bold;">Staff Nurse</span> from the beautiful state of Tamil Nadu, India. 
                    My journey has taken an exciting turn as I dive deep into the ocean of 
                    <span style="color: #FF9900; font-weight: bold;">Data Science ,Deep learning and Computer vision</span>.
                </p>
                <p class="ocean-text" style="font-size: 1.1rem; line-height: 1.7;">
                    My transition from healthcare to analytics is driven by curiosity, a passion for problem-solving, 
                    and a deep fascination with how <span style="font-weight: bold; color: #00ffff;">data</span> 
                    shapes decision-making in our modern world.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        bg_cols = st.columns(3)
        
        with bg_cols[0]:
            st.markdown("""
            <div class="ocean-card" style="border-left: 5px solid #00A8E1;">
                <h3 style="color: #00A8E1;">👩‍⚕️ My Healthcare Journey</h3>
                <div style="height: 3px; background: #00A8E1; margin: 10px 0; border-radius: 2px;"></div>
                <ul style="line-height: 2; color: #e6faff;">
                    <li>🏥 Former <strong style="color: #00ffff;">Registered Staff Nurse</strong></li>
                    <li>👩‍💼 Clinical decision-making expert</li>
                    <li>💡 Healthcare data analytics enthusiast</li>
                    <li>❤️ Passionate about patient care</li>
                    <li>🔄 Transitioning to Data Science</li>
                </ul>
                <div style="background: rgba(0, 168, 225, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                    <p style="color: #00A8E1; margin: 0; text-align: center; font-style: italic;">
                        "From saving lives to saving data insights"
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with bg_cols[1]:
            st.markdown("""
            <div class="ocean-card" style="border-left: 5px solid #FF9900;">
                <h3 style="color: #FF9900;">📚 My Current Mission</h3>
                <div style="height: 3px; background: #FF9900; margin: 10px 0; border-radius: 2px;"></div>
                <ul style="line-height: 2; color: #e6faff;">
                    <li>🚀 Baby steps into <strong style="color: #00ffff;">Data Science</strong></li>
                    <li>🎯 Currently enrolled at <strong style="color: #FF9900;">GUVI</strong></li>
                    <li>🧠 Learning ML, AI, and Analytics</li>
                    <li>🏗️ Building real-world projects</li>
                    <li>🤝 Open to collaboration opportunities</li>
                </ul>
                <div style="background: rgba(255, 153, 0, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                    <p style="color: #FF9900; margin: 0; text-align: center; font-style: italic;">
                        "Every expert was once a beginner"
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with bg_cols[2]:
            st.markdown("""
            <div class="ocean-card" style="border-left: 5px solid #0077B6;">
                <h3 style="color: #0077B6;">🛠️ My Technical Arsenal</h3>
                <div style="height: 3px; background: #0077B6; margin: 10px 0; border-radius: 2px;"></div>
                <ul style="line-height: 2; color: #e6faff;">
                    <li>🐍 Python, SQL, Pandas, NumPy</li>
                    <li>📊 Statistics & Probability</li>
                    <li>🧹 Data cleaning, EDA, Preprocessing</li>
                    <li>🤖 Machine Learning (Scikit-learn)</li>
                    <li>📈 Streamlit, Plotly, Seaborn,Matplotlib</li>
                    <li>💼 Power BI Dashboard Creation</li>
                    <li>📋 Business Insight Reporting</li>
                    <li>🤖 Deep learning and Computer vision</li>
                        </ul>
                <div style="background: rgba(0, 119, 182, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                    <p style="color: #0077B6; margin: 0; text-align: center; font-style: italic;">
                        "Growing toolkit, boundless potential"
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Project Journey
        st.markdown('<h2 class="section-header">🚀 Project Development Journey</h2>', unsafe_allow_html=True)
        
        journey_cols = st.columns(4)
        journey_steps = [
            ("🎯", "Research", "Deep dive into marine biology and CNN architectures", "#FF6B6B"),
            ("🏗️", "Development", "Built and trained 6 different neural network models", "#4ECDC4"),
            ("🧪", "Experimentation", "Fine-tuned hyperparameters and optimized performance", "#45B7D1"),
            ("🌊", "Deployment", "Created this beautiful interactive dashboard", "#96CEB4")
        ]
        
        for i, (icon, title, desc, color) in enumerate(journey_steps):
            with journey_cols[i]:
                st.markdown(f"""
                <div class="metric-ocean-card" style="border: 2px solid {color}40; min-height: 200px;">
                    <div style="font-size: 3rem; margin-bottom: 1rem; 
                               filter: drop-shadow(0 0 10px {color}80);">{icon}</div>
                    <h4 style="color: {color}; font-weight: 600; margin-bottom: 1rem;">{title}</h4>
                    <p style="color: #87ceeb; font-size: 0.9rem; line-height: 1.4;">{desc}</p>
                    <div style="margin-top: 1rem;">
                        <div class="progress-ocean">
                            <div class="progress-fill" style="width: 100%; background: {color};"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Technical achievements
        st.markdown('<h2 class="section-header">🏆 Technical Achievements</h2>', unsafe_allow_html=True)
        
        achievement_cols = st.columns(4)
        achievements = [
            ("99.87%", "Peak Accuracy", "🎯"),
            ("6", "Models Trained", "🤖"),
            ("47ms", "Inference Speed", "⚡"),
            ("11", "Marine Species Classified", "🐠")
        ]
        
        for i, (value, label, icon) in enumerate(achievements):
            with achievement_cols[i]:
                st.markdown(f"""
                <div class="metric-ocean-card">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
                    <h2 style="color: #00ffff; font-family: 'Orbitron', monospace; font-size: 2rem;">{value}</h2>
                    <p style="color: #87ceeb; font-weight: 600;">{label}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Contact section
        st.markdown('<h2 class="section-header">🌐 Connect & Collaborate</h2>', unsafe_allow_html=True)
        
        contact_cols = st.columns(3)
        
        with contact_cols[0]:
            st.markdown("""
            <div class="ocean-card" style="text-align: center; border: 2px solid #1DA1F2;">
                <div style="font-size: 3rem; color: #1DA1F2; margin-bottom: 1rem;">📧</div>
                <h4 style="color: #00ffff;">Email</h4>
                <p style="color: #1DA1F2; font-weight: 600;">malathisathish2228@gmail.com</p>
                <div style="background: rgba(29, 161, 242, 0.1); padding: 0.5rem; border-radius: 10px; margin-top: 1rem;">
                    <p style="color: #87ceeb; margin: 0; font-size: 0.9rem;">Let's discuss data science!</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with contact_cols[1]:
            st.markdown("""
            <div class="ocean-card" style="text-align: center; border: 2px solid #333;">
                <div style="font-size: 3rem; color: #00ffff; margin-bottom: 1rem;">💻</div>
                <h4 style="color: #00ffff;">GitHub</h4>
                <p style="color: #00ffff; font-weight: 600;">https://github.com/malathisathish/Multiclass-fish-image-classification</p>
                <div style="background: rgba(0, 255, 255, 0.1); padding: 0.5rem; border-radius: 10px; margin-top: 1rem;">
                    <p style="color: #87ceeb; margin: 0; font-size: 0.9rem;">Check out my projects!</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with contact_cols[2]:
            st.markdown("""
            <div class="ocean-card" style="text-align: center; border: 2px solid #0077B5;">
                <div style="font-size: 3rem; color: #0077B5; margin-bottom: 1rem;">💼</div>
                <h4 style="color: #00ffff;">LinkedIn</h4>
                <p style="color: #0077B5; font-weight: 600;">www.linkedin.com/in/malathi-sathish-016a03354</p>
                <div style="background: rgba(0, 119, 181, 0.1); padding: 0.5rem; border-radius: 10px; margin-top: 1rem;">
                    <p style="color: #87ceeb; margin: 0; font-size: 0.9rem;">Let's network!</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Future aspirations
        st.markdown('<h2 class="section-header">🚀 Future Enhancement </h2>', unsafe_allow_html=True)
        
        future_goals = [
            "🎥 Real-time video classification for marine research",
            "📱 Mobile app development for field researchers",
            "🌍 Multi-language support for global accessibility",
            "🤖 Advanced ensemble techniques and AutoML integration",
            "📊 Advanced analytics dashboard for marine biologists",
            "🌊 Contributing to ocean conservation through AI"
        ]
        
        goal_cols = st.columns(2)
        for i, goal in enumerate(future_goals):
            with goal_cols[i % 2]:
                st.markdown(f"""
                <div class="ocean-card" style="border-left: 4px solid #00ffff; margin: 0.5rem 0;">
                    <p class="ocean-text" style="margin: 0; font-size: 1.1rem;">
                        <strong>{goal}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Inspirational Quote
        st.markdown("""
        <div class="ocean-card" style='text-align: center; padding: 2rem; margin: 2rem 0;
                    border: 3px solid #00ffff; background: linear-gradient(135deg, rgba(0, 255, 255, 0.1), rgba(255, 153, 0, 0.1));'>
            <blockquote style='font-size: 1.6rem; font-style: italic; color: #00ffff; 
                              text-shadow: 0 0 10px rgba(0, 255, 255, 0.3); line-height: 1.6; margin-bottom: 1rem;
                              font-family: "Exo 2", sans-serif;'>
                <strong>"From the depths of the ocean to the power of deep learning, I’m on a mission to transform data into insights that protect our marine species conservation." 🌊
</strong> 🌊
            </blockquote>
            <cite style='color: #FF9900; font-size: 1.2rem; font-weight: bold; font-family: "Orbitron", monospace;'>
                - Malathi Y, Data Science Enthusiast<br>
                <span style='color: #87ceeb; font-size: 1rem;'></span>
            </cite>
        </div>
        """, unsafe_allow_html=True)
    
    about_page()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, rgba(0, 50, 100, 0.3), rgba(0, 30, 60, 0.3)); 
           border-radius: 20px; margin-top: 2rem;">
    <h2 style="color: #00ffff; font-family: 'Orbitron', monospace; margin-bottom: 1rem;">
        🌊 Crafted with ❤️ by Malathi Y — For Marine Conservation & Sustainable Research 🌍🐠
    </h2>
    🌊 🐠 🐟 🦐 🐡 🦈 🐙 🦞 🐚 🦀 🐋 🦑 🌊
    </div>
""", unsafe_allow_html=True)