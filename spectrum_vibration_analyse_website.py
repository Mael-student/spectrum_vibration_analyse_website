import streamlit as st
import numpy as np
import pandas as pd
import joblib
import io
import os

# ==========================================
# 1. SETUP & PROFESSIONAL STYLING
# ==========================================
st.set_page_config(page_title="Vibration Diagnostic System", layout="wide")

st.markdown("""
    <style>
    .reportview-container { background: #F5F7F9; }
    .main-title { font-size:32px; font-weight: 700; color: #1A365D; margin-bottom: 5px; }
    .sub-title { font-size:14px; color: #718096; margin-bottom: 30px; }
    .section-header { font-size:18px; font-weight: 600; color: #2D3748; margin: 25px 0 10px 0; border-bottom: 1px solid #E2E8F0; }
    .result-card { padding: 30px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 2px; }
    .stNumberInput > label { font-size: 14px; color: #4A5568; }
    .stAlert { border-radius: 2px; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 15px; border-radius: 4px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. TECHNICAL MAPPINGS (1 TO 7 STRICT MATCH)
# ==========================================
measurement_points_mapping = {
    "Motor Inboard Axial": 1, "Motor Inboard Horizontal": 2, "Motor Inboard Horz Peakvue": 3,
    "Motor Inboard Vertical": 4, "Motor Inboard X Probe": 5, "Motor Inboard Y Probe": 6,
    "Motor Outboard Axial": 7, "Motor Outboard Horizontal": 8, "Motor Outboard Horz Peakvue": 9,
    "Motor Outboard Vertical": 10, "Motor Outboard X Probe": 11, "Motor Outboard Y Probe": 12,
    "Pump Inboard Axial": 13, "Pump Inboard Horizontal": 14, "Pump Inboard Horz Peakvue": 15,
    "Pump Inboard Thrust 1 Hor": 16, "Pump Inboard Thrust 1 Ver": 17, "Pump Inboard Vertical": 18,
    "Pump Inboard X Probe": 19, "Pump Inboard Y Probe": 20, "Pump Outboard Axial": 21,
    "Pump Outboard Horizontal": 22, "Pump Outboard Horz Peakvue": 23, "Pump Outboard Vertical": 24,
    "Pump Outboard X Probe": 25, "Pump Outboard Y Probe": 26
}

fault_names = {
    1: 'Angular Misalignment', 2: 'Ball Defect', 3: 'Parallel Misalignment',
    4: 'Rotating Looseness', 5: 'Rotor Rub', 6: 'Turbulence', 7: 'Unbalance'
}

# --- Corrected Vibration Categorization ---
sub_synchronous_cols = ["0.1X-0.8X", "0.33X", "0.38X", "0.48X", "0.5X", "0.8X-1X"]
# 30X, 45X, and 80X are integer multiples, strictly placed under Synchronous (High Harmonics)
synchronous_cols = ["1X", "2X", "3X", "4X", "5X", "6X", "7X", "8X", "9X", "10X", "12X", "14X", "15X", "16X", "30X", "45X", "80X"]
non_synchronous_cols = ["1.5X", "1.9X", "2.5X", "3.5X", "3.84X", "4.16X", "4.2X", "5.9X", "6.3X", "9X-30X", "11.3X", "13.8X"]

# Preservation of the exact underlying feature order expected by the .joblib model
harmonics_columns = [
    "0.1X-0.8X", "0.33X", "0.38X", "0.48X", "0.5X", "0.8X-1X", "1X", "1.5X", "1.9X", "2X", 
    "2.5X", "3X", "3.5X", "3.84X", "4X", "4.16X", "4.2X", "5X", "5.9X", "6X", 
    "6.3X", "7X", "8X", "9X", "9X-30X", "10X", "11.3X", "12X", "13.8X", "14X", 
    "15X", "16X", "30X", "45X", "80X"
]

required_columns = ['MptDesc', 'RPM'] + harmonics_columns

# ==========================================
# 3. MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():
    model_filename = 'hgb_maintenance_model.joblib'
    if not os.path.exists(model_filename):
        st.error(f"❌ System Error: The file '{model_filename}' was not found in the current directory.")
        st.write("Files found in the script directory:", os.listdir('.'))
        return None
    try:
        return joblib.load(model_filename)
    except Exception as e:
        st.error(f"❌ Critical error occurred while loading the `.joblib` model:")
        st.exception(e)
        return None

model = load_model()

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "diagnostic"

st.sidebar.markdown("## 🧭 Navigation")

if st.sidebar.button("📊 Diagnostic Engine", use_container_width=True):
    st.session_state["current_page"] = "diagnostic"

if st.sidebar.button("📖 Technical Documentation", use_container_width=True):
    st.session_state["current_page"] = "documentation"


# ==========================================
# PAGE A: DIAGNOSTIC ENGINE
# ==========================================
if st.session_state["current_page"] == "diagnostic":
    st.markdown('<p class="main-title">Water Injection Pump Diagnostic System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Vibration Spectrum Analysis Engine (HistGradientBoosting)</p>', unsafe_allow_html=True)

    if not model:
        st.error("⚠️ **Model file not found (`hgb_maintenance_model.joblib`).** Please ensure the trained model file is placed in the exact same directory as this script.")

    analysis_mode = st.radio(
        "Select Data Input Method:",
        ["Manual Entry", "Excel / CSV File Upload"],
        horizontal=True
    )

    st.markdown("---")

    if analysis_mode == "Manual Entry":
        st.markdown('<p class="section-header">Asset Operational Context</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            selected_point_name = st.selectbox("Measurement Point Description (MptDesc)", list(measurement_points_mapping.keys()))
            mpt_desc = measurement_points_mapping[selected_point_name]
        with c2:
            rpm_val = st.number_input("Rotation Speed (RPM)", min_value=0.0, value=1500.0)

        st.markdown('<p class="section-header">Vibration Spectrum Magnitudes (In/Sec Pk)</p>', unsafe_allow_html=True)
        cols = st.columns(5)
        harmonic_inputs = []

        for i, label in enumerate(harmonics_columns):
            with cols[i % 5]:
                val = st.number_input(label, min_value=0.0, value=0.0, format="%.10f", step=0.0001, key=label)
                harmonic_inputs.append(val)

        st.markdown("---")
        if st.button("RUN SINGLE DIAGNOSTIC"):
            if model:
                features = [mpt_desc, rpm_val] + harmonic_inputs
                input_row_df = pd.DataFrame([features], columns=required_columns)
                
                pred = model.predict(input_row_df)[0]
                probabilities = model.predict_proba(input_row_df)[0]
                
                final_id = int(pred)
                diag = fault_names.get(final_id, "Normal Condition")
                general_confidence = np.max(probabilities) * 100
                
                st.markdown('<p class="section-header">Analysis Result</p>', unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="result-card">
                        <p style="color: #718096; text-transform: uppercase; letter-spacing: 1px; font-size: 12px;">Conclusion</p>
                        <h2 style="color: #1A365D; margin: 0;">{diag}</h2>
                        <p style="margin-top: 10px; font-size: 18px; color: #2B6CB0; font-weight: bold;">
                            Confidence Score: {general_confidence:.2f}%
                        </p>
                        <p style="margin-top: 10px; color: #4A5568;">
                            The system identified patterns consistent with <b>{diag.lower()}</b> at <b>{selected_point_name}</b>.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<p class="section-header">Probability Distribution Across All Fault Modes</p>', unsafe_allow_html=True)
                prob_dict = {fault_names[uid]: probabilities[i] * 100 for i, uid in enumerate(model.classes_)}
                df_probs = pd.DataFrame(list(prob_dict.items()), columns=["Failure Mode", "Probability (%)"])
                st.bar_chart(df_probs.set_index("Failure Mode"), horizontal=True)
            else:
                st.error("Cannot execute diagnostic. Model file not found.")

    else:
        st.markdown('<p class="section-header">Data Acquisition (File Import)</p>', unsafe_allow_html=True)
        with st.expander("📋 Batch Import File Requirements & Validation Specifications", expanded=True):
            st.markdown("""
            * **Supported Formats:** Standard Excel (`.xlsx`) or Comma-Separated Values (`.csv`).
            * **Multi-Row Batch Processing:** A single uploaded file may contain **multiple data entries**.
            * **Column Order Flexibility:** Columns can be arranged in **ANY** order. The system dynamically auto-aligns features.
            """)
            
            st.markdown("---")
            st.markdown("### 🔍 Required Columns and Dimensions Details")
            st.markdown("""
            Your file must contain exactly **37 variables** per row, matching the structural needs of the model:
            
            1. **`MptDesc`**: Textual context variable (e.g., *Motor Inboard Axial*, *Pump Outboard Vertical*, etc.).
            2. **`RPM`**: Kinematic operational variable (Rotational speed).
            3. **The 35 Target Spectral Amplitudes (Physical magnitudes in *In/Sec Pk*) grouped by types:**
            """)
            
            doc_col1, doc_col2, doc_col3 = st.columns(3)
            with doc_col1:
                st.markdown("**Sub-Synchronous Bands:**")
                st.code("\n".join(sub_synchronous_cols), language="text")
            with doc_col2:
                st.markdown("**Synchronous (Fundamental & Harmonics):**")
                st.code("\n".join(synchronous_cols), language="text")
            with doc_col3:
                st.markdown("**Non-Synchronous Bands:**")
                st.code("\n".join(non_synchronous_cols), language="text")

        uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'csv'])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_input = pd.read_csv(uploaded_file)
                else:
                    df_input = pd.read_excel(uploaded_file)

                st.markdown('<p class="section-header">Data Preview</p>', unsafe_allow_html=True)
                st.dataframe(df_input.head(5), use_container_width=True)

                missing_cols = [c for c in required_columns if c not in df_input.columns]
                
                if missing_cols:
                    st.error(f"Invalid file structure. Missing columns: {', '.join(missing_cols)}")
                else:
                    if st.button("RUN BATCH DIAGNOSTIC"):
                        if model:
                            predictions_list = []
                            all_general_confidences = []
                            fault_confidence_groups = {fault_names[uid]: [] for uid in model.classes_}
                            
                            for i in range(len(df_input)):
                                row_data = df_input.iloc[i]
                                text_mpt = str(row_data['MptDesc']).strip()
                                
                                if text_mpt in measurement_points_mapping:
                                    numeric_mpt = measurement_points_mapping[text_mpt]
                                    features = [numeric_mpt, float(row_data['RPM'])] + [float(row_data[h]) for h in harmonics_columns]
                                    input_row_df = pd.DataFrame([features], columns=required_columns)
                                    
                                    pred = model.predict(input_row_df)[0]
                                    probabilities = model.predict_proba(input_row_df)[0]
                                    
                                    final_id = int(pred)
                                    diag = fault_names.get(final_id, "Normal Condition")
                                    conf_gen = np.max(probabilities) * 100
                                    
                                    if diag in fault_confidence_groups:
                                        fault_confidence_groups[diag].append(conf_gen)
                                else:
                                    diag = "❌ Invalid MptDesc"
                                    conf_gen = 0.0

                                predictions_list.append(diag)
                                all_general_confidences.append(conf_gen)

                            df_main_results = df_input[required_columns].copy()
                            df_main_results.insert(0, "Diagnostic Result", predictions_list)

                            summary_data = []
                            for uid in model.classes_:
                                f_name = fault_names[uid]
                                scores = fault_confidence_groups[f_name]
                                count = len(scores)
                                mean_confidence = np.mean(scores) if count > 0 else 0.0
                                
                                summary_data.append({
                                    "Analysis Metric": f"Average Confidence: {f_name}",
                                    "Detected Instances Count": count,
                                    "Value (%)": round(mean_confidence, 2)
                                })
                            
                            global_general_conf = np.mean(all_general_confidences) if len(all_general_confidences) > 0 else 0.0
                            summary_data.append({
                                "Analysis Metric": "TOTAL GENERAL CONFIDENCE SCORE (OVERALL LOT)",
                                "Detected Instances Count": len(df_input),
                                "Value (%)": round(global_general_conf, 2)
                            })
                            
                            df_global_summary = pd.DataFrame(summary_data)

                            st.markdown('<p class="section-header">Automated Diagnostic Report</p>', unsafe_allow_html=True)
                            
                            b_col1, b_col2, _ = st.columns([1, 1, 3])
                            with b_col1:
                                csv_data = df_main_results.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Diagnostics (CSV)",
                                    data=csv_data,
                                    file_name="vibration_diagnostic_report.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            with b_col2:
                                buffer = io.BytesIO()
                                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                    df_main_results.to_excel(writer, index=False, sheet_name='Diagnostics')
                                    df_global_summary.to_excel(writer, index=False, sheet_name='Global Summary')
                                buffer.seek(0)
                                st.download_button(
                                    label="📊 Download Multi-Sheet Excel (XLSX)",
                                    data=buffer,
                                    file_name="vibration_diagnostic_report.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            
                                st.markdown("### 🖥️ On-Screen Report Preview")
                            tab1, tab2 = st.tabs(["📊 Main Diagnostics Table (Ordered)", "🎯 Lot Confidence Summary"])
                            with tab1:
                                st.dataframe(df_main_results, use_container_width=True)
                            with tab2:
                                st.dataframe(df_global_summary, use_container_width=True)
                        else:
                            st.error("Cannot execute diagnostic. Model file not found.")
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

# ==========================================
# PAGE B: TECHNICAL DOCUMENTATION
# ==========================================
else:
    st.markdown('<p class="main-title">📖 Model Documentation & Technical Specs</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Scope of application, architecture, and real performance metrics obtained during test phase</p>', unsafe_allow_html=True)
    
    # --- a) MODEL LIMITATIONS ---
    st.markdown('<p class="section-header">a) Technical Specifications & Model Limitations</p>', unsafe_allow_html=True)
    st.info("""
    * **Technological Restriction:** This model is strictly limited and tuned to the specific mechanical impedance and behavioral dynamics of **water injection pumps**.
    * **Descriptor Dependency:** The application strictly requires the presence of all 35 target harmonics. Raw, un-binned vibration frequency spectrums cannot be processed directly.
    * **Speed Range (RPM):** The accuracy of the results heavily depends on the operational conformity of the inputted rotational speeds relative to the limits established in the baseline training dataset.
    """)

    # --- b) DATASET DETAILS ---
    st.markdown('<p class="section-header">b) Dataset Details</p>', unsafe_allow_html=True)
    st.markdown("""
    The baseline archive used to fit and extract features contains high-fidelity engineering parameters:
    
    * **Baseline Dataset Volume:** **~700 lines** of real historical vibration records.
    * **Dimensionality:** **37 input parameters**:
        * `1` textual contextual variable (`MptDesc` converted to numerical IDs from 1 to 26).
        * `1` kinematic operational variable (rotational speed in `RPM`).
        * `35` spectral variables (physical amplitudes measured in *In/Sec Pk* over specific frequency bands).
    
    **Complete List of the 35 Target Spectral Amplitudes grouped by types:**
    """)
    
    doc_col1, doc_col2, doc_col3 = st.columns(3)
    with doc_col1:
        st.markdown("**Sub-Synchronous Bands:**")
        st.code("\n".join(sub_synchronous_cols), language="text")
    with doc_col2:
        st.markdown("**Synchronous (Fundamental & Harmonics):**")
        st.code("\n".join(synchronous_cols), language="text")
    with doc_col3:
        st.markdown("**Non-Synchronous Bands:**")
        st.code("\n".join(non_synchronous_cols), language="text")

    # --- c) MODEL ARCHITECTURE ---
    st.markdown('<p class="section-header">c) Model Architecture (How the Model Works)</p>', unsafe_allow_html=True)
    st.markdown("""
    The artificial intelligence built into this system utilizes an advanced Machine Learning algorithm called **HistGradientBoosting** (Histogram-Based Gradient Boosting Machine) to compute diagnostics:
    
    1. **Data Discretization (Histogram Binning):** Continuous numerical inputs (the 35 vibration amplitudes) are grouped into 256 integer bins. This dramatically reduces memory usage and speeds up tree-building computations.
    2. **Ensemble of Decision Trees:** The algorithm sequentially builds multiple shallow decision trees. Each subsequent tree is strictly trained to minimize and correct the prediction errors (residuals) made by the previous ones.
    3. **Pattern Mapping:** By analyzing cross-harmonic interactions (e.g., matching a simultaneous spike at 1X and 2X), the model evaluates the mathematical probability for each of the 7 failure categories and outputs the highest score as the final diagnostic conclusion.
    """)
    
    # --- d) EVALUATION RESULTS ---
    st.markdown('<p class="section-header">d) Evaluation Results (Final Test Set)</p>', unsafe_allow_html=True)
    st.markdown("Here are the general metrics calculated during evaluation on the external evaluation batch (*Test Set*):")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-card"><p style="margin:0;font-size:12px;color:#718096;">TEST SET ACCURACY</p><h2 style="margin:0;color:#2B6CB0;">82.28%</h2></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><p style="margin:0;font-size:12px;color:#718096;">TEST VOLUME (SUPPORT)</p><h2 style="margin:0;color:#2B6CB0;">79 Rows</h2></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><p style="margin:0;font-size:12px;color:#718096;">MACRO AVG F1-SCORE</p><h2 style="margin:0;color:#2B6CB0;">0.85</h2></div>', unsafe_allow_html=True)
        
    st.markdown("#### 📋 Detailed Classification Report by Failure Mode")
    
    report_data = {
        "ID": [1, 2, 3, 4, 5, 6, 7],
        "Failure Mode Name": [
            "Angular Misalignment (1)", "Ball Defect (2)", "Parallel Misalignment (3)", 
            "Rotating Looseness (4)", "Rotor Rub (5)", "Turbulence (6)", "Unbalance (7)"
        ],
        "Precision": [1.00, 0.80, 0.75, 0.82, 1.00, 0.70, 0.86],
        "Recall": [1.00, 1.00, 0.75, 0.50, 1.00, 0.93, 0.89],
        "F1-Score": [1.00, 0.89, 0.75, 0.62, 1.00, 0.80, 0.87],
        "Support (Test Rows)": [6, 4, 4, 18, 5, 15, 27]
    }
    df_report = pd.DataFrame(report_data)
    st.dataframe(df_report.set_index("ID"), use_container_width=True)

# ==========================================
# 7. FOOTER / SYSTEM INFO
# ==========================================
st.sidebar.markdown("---")
st.sidebar.caption("GIM Maintenance Hub - v4.10")
st.sidebar.caption("HistGradientBoosting Engine")
