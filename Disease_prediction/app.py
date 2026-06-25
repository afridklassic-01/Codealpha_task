import os
import pickle
import numpy as np
import streamlit as st

# Set web page configuration
st.set_page_config(
    page_title="Health Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS Styling
st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: 700; color: #1E3A8A; margin-bottom: 5px; }
    .subtitle { font-size: 16px; color: #6B7280; margin-bottom: 30px; }
    .report-card { padding: 20px; border-radius: 10px; margin-top: 20px; }
    .positive-risk { background-color: #FEE2E2; border-left: 5px solid #EF4444; color: #991B1B; }
    .negative-risk { background-color: #D1FAE5; border-left: 5px solid #10B981; color: #065F46; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts(label):
    """Dynamically loads and caches the trained ML models and scalers."""
    model_path = f'models/{label}_model.pkl'
    scaler_path = f'models/{label}_scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    with open(model_path, 'rb') as m_file:
        model = pickle.load(m_file)
    with open(scaler_path, 'rb') as s_file:
        scaler = pickle.load(s_file)
        
    return model, scaler

# Sidebar Navigation Panel (Updated to use native emoji heading instead of external image URL)
st.sidebar.markdown("# 🩺 Clinical Dashboard")
st.sidebar.title("Diagnostic Center")
st.sidebar.markdown("Select a specialized clinical predictive panel below.")

panel_selection = st.sidebar.radio(
    "Select Diagnostic Panel",
    ["Diabetes Screening", "Cardiovascular Evaluation", "Oncology Classification"]
)

# Core App Workspace Mapping
if panel_selection == "Diabetes Screening":
    st.markdown('<div class="main-title">Diabetes Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predicts diabetes susceptibility based on Pima Indian diagnostic criteria.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Plasma Glucose Concentration (2-hr test)", min_value=0.0, max_value=300.0, value=120.0)
        bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0.0, max_value=150.0, value=70.0)
        skin = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
    with col2:
        insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.4)
        dpf = st.number_input("Diabetes Pedigree Function Metric", min_value=0.0, max_value=3.0, value=0.47, format="%.3f")
        age = st.number_input("Patient Age (Years)", min_value=1, max_value=120, value=33)
        
    raw_features = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
    classes = ["Low Risk / Negative", "High Risk / Positive"]
    label_key = "diabetes"

elif panel_selection == "Cardiovascular Evaluation":
    st.markdown('<div class="main-title">Cardiovascular Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyzes clinical features using the Cleveland Heart Disease metrics.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Patient Age (Years)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Biological Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.slider("Chest Pain Type (cp score)", min_value=0, max_value=3, value=1)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")
        restecg = st.slider("Resting ECG Results", min_value=0, max_value=2, value=0)
    with col2:
        thalch = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
        slope = st.slider("Slope of Peak Exercise ST Segment", min_value=1, max_value=3, value=1)
        ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
        thal = st.slider("Thalassemia Structural Score (1=Normal, 2=Fixed, 3=Reversible)", min_value=1, max_value=3, value=2)
        
    raw_features = [age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]
    classes = ["No Heart Disease Detected", "Heart Disease Indicators Present"]
    label_key = "heart"

else:
    st.markdown('<div class="main-title">Oncology Classification Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Classifies cellular mass characteristics derived from digitized fine needle aspirates.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0, value=14.0)
        texture_mean = st.number_input("Texture Mean", min_value=0.0, value=19.0)
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=92.0)
        area_mean = st.number_input("Area Mean", min_value=0.0, value=650.0)
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.096, format="%.4f")
    with col2:
        compactness_mean = st.number_input("Compactness Mean", min_value=0.0, value=0.104, format="%.4f")
        concavity_mean = st.number_input("Concavity Mean", min_value=0.0, value=0.088, format="%.4f")
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, value=0.048, format="%.4f")
        symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, value=0.181, format="%.4f")
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, value=0.062, format="%.4f")
        
    base_inputs = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                   compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean]
    raw_features = base_inputs + [0.0] * 20
    classes = ["Benign (Non-Cancerous Cell Structure)", "Malignant (Cancerous Cell Structure)"]
    label_key = "general"

st.markdown("---")

# Execution Pipeline Action
if st.button("Generate Diagnostic Report", type="primary"):
    model, scaler = load_artifacts(label_key)
    
    if model is None or scaler is None:
        st.error(f"Missing serialization artifacts for {label_key}. Run train_model.py inside terminal first.")
    else:
        try:
            # Scale and Predict
            vector = np.array([raw_features])
            scaled_vector = scaler.transform(vector)
            prediction = model.predict(scaled_vector)[0]
            
            # Formulate textual output
            is_positive = (prediction == 1 or prediction == '1')
            result_str = classes[1] if is_positive else classes[0]
            
            st.subheader("📋 Evaluation Summary")
            if is_positive:
                st.markdown(f'<div class="report-card positive-risk"><strong>DIAGNOSIS STATUS:</strong> {result_str}<br><br><em>Action Plan: Immediate medical follow-up recommended.</em></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="report-card negative-risk"><strong>DIAGNOSIS STATUS:</strong> {result_str}<br><br><em>Action Plan: Diagnostic parameters currently check out within regular reference ranges.</em></div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Inference execution fault: {e}")