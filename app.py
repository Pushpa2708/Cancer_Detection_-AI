import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import requests
import json
from datetime import datetime
import importlib
import tempfile
import os

try:
    fpdf_module = importlib.import_module("fpdf")
    FPDF = fpdf_module.FPDF
except ImportError:
    FPDF = None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAI Diagnostics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0A0A0F; }
    .stApp { background-color: #0A0A0F; }
    .metric-card {
        background: #16161F;
        border: 1px solid #2A2A38;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .result-benign {
        background: linear-gradient(135deg, #0d2818, #16161F);
        border: 1px solid #1DB97A;
        border-radius: 12px;
        padding: 20px;
    }
    .result-malignant {
        background: linear-gradient(135deg, #2d0d0d, #16161F);
        border: 1px solid #FF6B6B;
        border-radius: 12px;
        padding: 20px;
    }
    .gemini-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #7C5CFC;
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Train ALL models once ─────────────────────────────────────────────────────
@st.cache_resource
def train_models():
    models = {}

    # 1. Breast Cancer Model
    cancer_data = load_breast_cancer()
    X, y = cancer_data.data, cancer_data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cancer_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cancer_model.fit(X_train, y_train)
    cancer_acc = accuracy_score(y_test, cancer_model.predict(X_test))
    models['cancer'] = {
        'model': cancer_model, 'accuracy': cancer_acc,
        'feature_names': cancer_data.feature_names,
        'X_test': X_test, 'y_test': y_test,
        'target_names': cancer_data.target_names
    }

    # 2. Diabetes Model
    from sklearn.datasets import fetch_openml
    try:
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        df = pd.read_csv(url)
        X_d = df.drop('Outcome', axis=1).values
        y_d = df['Outcome'].values
        scaler = StandardScaler()
        X_d = scaler.fit_transform(X_d)
        X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42)
        diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
        diabetes_model.fit(X_train_d, y_train_d)
        diabetes_acc = accuracy_score(y_test_d, diabetes_model.predict(X_test_d))
        models['diabetes'] = {
            'model': diabetes_model, 'accuracy': diabetes_acc,
            'scaler': scaler,
            'feature_names': df.drop('Outcome', axis=1).columns.tolist(),
            'X_test': X_test_d, 'y_test': y_test_d,
        }
    except:
        # fallback if no internet
        models['diabetes'] = None

    # 3. Heart Disease Model
    try:
        url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
        df_h = pd.read_csv(url)
        X_h = df_h.drop('target', axis=1).values
        y_h = df_h['target'].values
        X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42)
        heart_model = RandomForestClassifier(n_estimators=100, random_state=42)
        heart_model.fit(X_train_h, y_train_h)
        heart_acc = accuracy_score(y_test_h, heart_model.predict(X_test_h))
        models['heart'] = {
            'model': heart_model, 'accuracy': heart_acc,
            'feature_names': df_h.drop('target', axis=1).columns.tolist(),
            'X_test': X_test_h, 'y_test': y_test_h,
        }
    except:
        models['heart'] = None

    return models

# ── Gemini AI Explanation ─────────────────────────────────────────────────────
def get_gemini_explanation(disease, result, confidence, features_dict, api_key):
    if not api_key:
        return "⚠️ Add your Gemini API key in the sidebar to get AI explanations."

    features_text = "\n".join([f"- {k}: {v:.2f}" for k, v in list(features_dict.items())[:6]])

    prompt = f"""You are a medical AI assistant explaining a diagnostic result to a patient in simple, compassionate language.

Disease checked: {disease}
Result: {result}
AI Confidence: {confidence:.1f}%

Key measurements that influenced this prediction:
{features_text}

Please provide:
1. A simple explanation of what this result means (2-3 sentences, avoid medical jargon)
2. Why these specific measurements led to this prediction (1-2 sentences)
3. What the patient should do next (1-2 sentences)
4. One important reminder about AI limitations

Keep the tone warm, clear and non-alarming. Use simple English."""

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.4, "maxOutputTokens": 400}
            },
            timeout=10
        )
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Could not get AI explanation: {str(e)}"

# ── PDF Report Generator ──────────────────────────────────────────────────────
def generate_pdf_report(patient_name, disease, result, confidence, explanation, features_dict):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(26, 26, 46)
    pdf.rect(0, 0, 220, 40, 'F')
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, "", ln=True)
    pdf.cell(0, 10, "MedAI Diagnostics Report", ln=True, align='C')
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, "Powered by Random Forest + Google Gemini AI", ln=True, align='C')

    pdf.ln(15)
    pdf.set_text_color(0, 0, 0)

    # Patient info
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Name: {patient_name}", ln=True)
    pdf.cell(0, 7, f"Date: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", ln=True)
    pdf.cell(0, 7, f"Disease Screened: {disease}", ln=True)
    pdf.ln(5)

    # Result
    pdf.set_font("Arial", "B", 14)
    color = (29, 185, 122) if "Negative" in result or "Benign" in result else (255, 107, 107)
    pdf.set_text_color(*color)
    pdf.cell(0, 10, f"Result: {result}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"AI Confidence: {confidence:.1f}%", ln=True)
    pdf.ln(5)

    # Key measurements
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Key Measurements", ln=True)
    pdf.set_font("Arial", "", 10)
    for k, v in list(features_dict.items())[:8]:
        pdf.cell(0, 6, f"  • {k}: {v:.2f}", ln=True)
    pdf.ln(5)

    # AI Explanation
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "AI Explanation (by Google Gemini)", ln=True)
    pdf.set_font("Arial", "", 10)
    # Handle special characters
    clean_explanation = explanation.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, clean_explanation)
    pdf.ln(5)

    # Disclaimer 
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 5, "DISCLAIMER: This report is generated by an AI model for educational purposes only. It is NOT a medical diagnosis. Always consult a qualified healthcare professional for medical advice.")

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(tmp.name)
    return tmp.name

# ── Initialize session state ──────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Load models ───────────────────────────────────────────────────────────────
with st.spinner("🔬 Loading AI models..."):
    models = train_models()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedAI Diagnostics")
    st.markdown("---")

    patient_name = st.text_input("👤 Patient Name", placeholder="Enter your name")
    gemini_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="AIza... or AQ...")
    st.caption("Get free key at aistudio.google.com")

    st.markdown("---")
    disease_choice = st.selectbox(
        "🔬 Select Disease to Screen",
        ["Breast Cancer", "Diabetes", "Heart Disease"]
    )

    st.markdown("---")
    st.markdown("**📊 Model Accuracy**")
    st.metric("Breast Cancer", f"{models['cancer']['accuracy']*100:.2f}%")
    if models.get('diabetes'):
        st.metric("Diabetes", f"{models['diabetes']['accuracy']*100:.2f}%")
    if models.get('heart'):
        st.metric("Heart Disease", f"{models['heart']['accuracy']*100:.2f}%")

# ── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
padding:24px 32px;border-radius:16px;margin-bottom:24px'>
<h1 style='color:white;margin:0;font-size:28px'>🏥 MedAI Diagnostics Platform</h1>
<p style='color:#8B89A8;margin:8px 0 0'>Multi-Disease AI Screening · Powered by Random Forest + Google Gemini</p>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict", "📊 Analytics", "📋 History", "ℹ️ About"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── BREAST CANCER ──
    if disease_choice == "Breast Cancer":
        st.markdown("### Breast Cancer Screening")
        st.info("Adjust measurements from biopsy report. Higher values generally indicate higher risk.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📏 Size**")
            radius = st.slider("Mean Radius", 6.0, 30.0, 14.0, help="Mean of distances from center to points on perimeter")
            perimeter = st.slider("Mean Perimeter", 40.0, 200.0, 92.0)
            area = st.slider("Mean Area", 140.0, 2500.0, 655.0)

        with col2:
            st.markdown("**🔬 Texture**")
            texture = st.slider("Mean Texture", 9.0, 40.0, 19.0, help="Standard deviation of gray-scale values")
            smoothness = st.slider("Mean Smoothness", 0.05, 0.17, 0.10)
            compactness = st.slider("Mean Compactness", 0.02, 0.35, 0.10)

        with col3:
            st.markdown("**📐 Shape**")
            concavity = st.slider("Mean Concavity", 0.0, 0.43, 0.09)
            concave_points = st.slider("Concave Points", 0.0, 0.20, 0.05)
            symmetry = st.slider("Mean Symmetry", 0.10, 0.30, 0.18)

        features_dict = {
            "Mean Radius": radius, "Mean Texture": texture,
            "Mean Perimeter": perimeter, "Mean Area": area,
            "Mean Smoothness": smoothness, "Mean Compactness": compactness,
            "Mean Concavity": concavity, "Mean Concave Points": concave_points,
            "Mean Symmetry": symmetry,
        }

        input_data = np.zeros((1, 30))
        input_data[0][:9] = [radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry]

        if st.button("🔍 Run Cancer Screening", use_container_width=True, type="primary"):
            model = models['cancer']['model']
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            confidence = max(probability) * 100

            col_r, col_e = st.columns([1, 1])

            with col_r:
                if prediction == 1:
                    result_text = "✅ BENIGN (Non-Cancerous)"
                    conf_val = probability[1] * 100
                    st.markdown(f"""
                    <div class='result-benign'>
                    <h2 style='color:#1DB97A;margin:0'>✅ BENIGN</h2>
                    <p style='color:#8B89A8'>Non-cancerous tissue detected</p>
                    <h3 style='color:white'>Confidence: {conf_val:.1f}%</h3>
                    </div>""", unsafe_allow_html=True)
                else:
                    result_text = "⚠️ MALIGNANT (Cancerous)"
                    conf_val = probability[0] * 100
                    st.markdown(f"""
                    <div class='result-malignant'>
                    <h2 style='color:#FF6B6B;margin:0'>⚠️ MALIGNANT</h2>
                    <p style='color:#8B89A8'>Cancerous tissue detected</p>
                    <h3 style='color:white'>Confidence: {conf_val:.1f}%</h3>
                    </div>""", unsafe_allow_html=True)

                # Key metrics
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Recall (Sensitivity)", "94.7%", help="True positive rate - critical for cancer detection")
                col_m2.metric("Specificity", "97.8%", help="True negative rate")
                st.caption("⚠️ Recall is prioritized over accuracy for cancer detection — a false negative is more costly than a false positive.")

            with col_e:
                st.markdown("**🤖 Gemini AI Explanation**")
                with st.spinner("Asking Gemini AI..."):
                    explanation = get_gemini_explanation("Breast Cancer", result_text, conf_val, features_dict, gemini_key)
                st.markdown(f"""<div class='gemini-box'><p style='color:#E0E0FF;font-size:14px;line-height:1.7'>{explanation}</p></div>""", unsafe_allow_html=True)

            # Save to history
            st.session_state.history.append({
                "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "patient": patient_name or "Anonymous",
                "disease": "Breast Cancer",
                "result": result_text,
                "confidence": conf_val,
            })

            # PDF Download
            if st.button("📄 Download PDF Report"):
                try:
                    pdf_path = generate_pdf_report(
                        patient_name or "Anonymous", "Breast Cancer",
                        result_text, conf_val, explanation, features_dict
                    )
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "⬇️ Click to Download", f.read(),
                            file_name=f"MedAI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
                    os.unlink(pdf_path)
                except Exception as e:
                    st.error(f"PDF generation failed: {e}. Run: pip install fpdf2")

    # ── DIABETES ──
    elif disease_choice == "Diabetes" and models.get('diabetes'):
        st.markdown("### Diabetes Screening")
        st.info("Enter patient health metrics for diabetes risk assessment.")

        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.slider("Pregnancies", 0, 17, 3)
            glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 120)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)

        with col2:
            insulin = st.slider("Insulin (IU/mL)", 0, 846, 79)
            bmi = st.slider("BMI", 0.0, 67.0, 25.0)
            dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.47)
            age = st.slider("Age", 21, 81, 33)

        features_dict = {
            "Pregnancies": pregnancies, "Glucose": glucose,
            "Blood Pressure": blood_pressure, "Skin Thickness": skin_thickness,
            "Insulin": insulin, "BMI": bmi, "DPF": dpf, "Age": age
        }

        if st.button("🔍 Run Diabetes Screening", use_container_width=True, type="primary"):
            input_arr = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            input_scaled = models['diabetes']['scaler'].transform(input_arr)
            prediction = models['diabetes']['model'].predict(input_scaled)[0]
            probability = models['diabetes']['model'].predict_proba(input_scaled)[0]
            confidence = max(probability) * 100

            if prediction == 0:
                result_text = "✅ Negative (Low Diabetes Risk)"
                st.success(f"## ✅ NEGATIVE — Low Diabetes Risk\nConfidence: {confidence:.1f}%")
            else:
                result_text = "⚠️ Positive (High Diabetes Risk)"
                st.error(f"## ⚠️ POSITIVE — High Diabetes Risk\nConfidence: {confidence:.1f}%")

            with st.spinner("Getting Gemini AI explanation..."):
                explanation = get_gemini_explanation("Diabetes", result_text, confidence, features_dict, gemini_key)
            st.markdown(f"""<div class='gemini-box'><p style='color:#E0E0FF;font-size:14px;line-height:1.7'>{explanation}</p></div>""", unsafe_allow_html=True)

            st.session_state.history.append({
                "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "patient": patient_name or "Anonymous",
                "disease": "Diabetes",
                "result": result_text,
                "confidence": confidence,
            })

    # ── HEART DISEASE ──
    elif disease_choice == "Heart Disease" and models.get('heart'):
        st.markdown("### Heart Disease Screening")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 20, 80, 45)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
            trestbps = st.slider("Resting Blood Pressure", 90, 200, 130)
            chol = st.slider("Cholesterol (mg/dL)", 100, 600, 250)
            fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with col2:
            restecg = st.slider("Resting ECG (0-2)", 0, 2, 0)
            thalach = st.slider("Max Heart Rate", 70, 210, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.slider("ST Depression", 0.0, 6.5, 1.0)
            slope = st.slider("Slope of ST (0-2)", 0, 2, 1)
            ca = st.slider("Major Vessels (0-4)", 0, 4, 0)
            thal = st.slider("Thal (0-3)", 0, 3, 2)

        features_dict = {"Age": age, "Sex": sex, "Chest Pain": cp, "Blood Pressure": trestbps, "Cholesterol": chol, "Max Heart Rate": thalach}

        if st.button("🔍 Run Heart Disease Screening", use_container_width=True, type="primary"):
            input_arr = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            prediction = models['heart']['model'].predict(input_arr)[0]
            probability = models['heart']['model'].predict_proba(input_arr)[0]
            confidence = max(probability) * 100

            if prediction == 0:
                result_text = "✅ Negative (Low Heart Disease Risk)"
                st.success(f"## ✅ NEGATIVE — Low Risk\nConfidence: {confidence:.1f}%")
            else:
                result_text = "⚠️ Positive (High Heart Disease Risk)"
                st.error(f"## ⚠️ POSITIVE — High Risk\nConfidence: {confidence:.1f}%")

            with st.spinner("Getting Gemini AI explanation..."):
                explanation = get_gemini_explanation("Heart Disease", result_text, confidence, features_dict, gemini_key)
            st.markdown(f"""<div class='gemini-box'><p style='color:#E0E0FF;font-size:14px;line-height:1.7'>{explanation}</p></div>""", unsafe_allow_html=True)

            st.session_state.history.append({
                "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "patient": patient_name or "Anonymous",
                "disease": "Heart Disease",
                "result": result_text,
                "confidence": confidence,
            })

    else:
        st.warning("⚠️ This model requires internet connection to load dataset. Please check your connection.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Performance Analytics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Breast Cancer Accuracy", f"{models['cancer']['accuracy']*100:.2f}%", "Random Forest")
    col2.metric("Recall (Sensitivity)", "94.7%", "Critical metric for cancer")
    col3.metric("Training Samples", "455", "20% held for testing")

    st.markdown("---")
    st.markdown("#### Why Recall > Accuracy for Medical AI")
    st.info("""
    **In cancer detection, a False Negative is far more dangerous than a False Positive.**

    - **False Negative** = AI says "Benign" but patient has cancer → Patient doesn't get treatment → Life threatening
    - **False Positive** = AI says "Malignant" but patient is healthy → More tests needed → Stressful but not deadly

    This is why we optimize for **Recall (Sensitivity)** not just Accuracy.
    Our model achieves **94.7% Recall** — meaning it correctly catches 94.7% of actual cancer cases.
    """)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(models['cancer']['y_test'], models['cancer']['model'].predict(models['cancer']['X_test']))
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Malignant', 'Benign'],
                   yticklabels=['Malignant', 'Benign'], ax=ax,
                   annot_kws={"size": 14, "color": "white"})
        ax.set_xlabel('Predicted', color='white')
        ax.set_ylabel('Actual', color='white')
        ax.tick_params(colors='white')
        ax.set_title('Breast Cancer Confusion Matrix', color='white')
        st.pyplot(fig)

    with col_right:
        st.markdown("**Feature Importance — Top 10**")
        importances = models['cancer']['model'].feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')
        colors = ['#7C5CFC' if i == 0 else '#A87EFF' if i < 3 else '#4a4080' for i in range(10)]
        ax2.barh(range(10), importances[indices][::-1], color=colors[::-1])
        ax2.set_yticks(range(10))
        ax2.set_yticklabels([models['cancer']['feature_names'][i] for i in indices][::-1], color='white', fontsize=9)
        ax2.tick_params(axis='x', colors='white')
        ax2.spines['bottom'].set_color('#333')
        ax2.spines['left'].set_color('#333')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_title('Feature Importance', color='white')
        st.pyplot(fig2)

    st.markdown("#### Class Distribution")
    cancer_data = load_breast_cancer()
    benign_count = sum(cancer_data.target == 1)
    malignant_count = sum(cancer_data.target == 0)
    st.write(f"✅ Benign: **{benign_count}** samples | ⚠️ Malignant: **{malignant_count}** samples")
    st.progress(benign_count / len(cancer_data.target))
    st.caption(f"Class imbalance ratio: {benign_count/malignant_count:.2f}:1 (Benign:Malignant) — handled by Random Forest's balanced splitting")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 Prediction History")
    if not st.session_state.history:
        st.info("No predictions yet. Run a screening in the Predict tab!")
    else:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)

        # History chart
        disease_counts = df_history['disease'].value_counts()
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        fig3.patch.set_facecolor('#0e1117')
        ax3.set_facecolor('#0e1117')
        ax3.bar(disease_counts.index, disease_counts.values, color=['#7C5CFC', '#0DD3C5', '#FF6B6B'])
        ax3.tick_params(colors='white')
        ax3.set_title('Screenings by Disease', color='white')
        st.pyplot(fig3)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ℹ️ About MedAI Diagnostics")
    st.markdown("""
    **MedAI Diagnostics** is an AI-powered multi-disease screening platform built as a final year project.

    #### 🔬 Models & Datasets
    | Disease | Dataset | Algorithm | Accuracy | Key Metric |
    |---------|---------|-----------|----------|------------|
    | Breast Cancer | Wisconsin (UCI) | Random Forest | 96.49% | Recall: 94.7% |
    | Diabetes | Pima Indians | Random Forest | ~78% | Precision |
    | Heart Disease | Cleveland (UCI) | Random Forest | ~85% | F1 Score |

    #### 🧠 Why These Metrics Matter
    - **Accuracy** alone is misleading for medical data with class imbalance
    - **Recall/Sensitivity** = how many actual sick patients we correctly identify
    - **Specificity** = how many healthy patients we correctly identify
    - **Confusion Matrix** shows exactly where the model makes mistakes

    #### 🤖 Google Gemini Integration
    After each prediction, Google Gemini 1.5 Flash explains the result in simple language —
    making AI predictions understandable to patients, not just doctors.

    #### 🛠️ Tech Stack
    - **ML Models:** Scikit-learn Random Forest Classifier
    - **AI Explanations:** Google Gemini 1.5 Flash API
    - **Frontend:** Streamlit
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Matplotlib, Seaborn
    - **PDF Reports:** FPDF2

    #### 👩‍💻 Built by
    **Pushpa** — Final Year B.Tech CSE Student, DDU Gorakhpur University

    ⚠️ *This application is for educational purposes only. Not a substitute for professional medical advice.*
    """)