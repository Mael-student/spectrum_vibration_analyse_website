import streamlit as st
import numpy as np
import pandas as pd
import joblib

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
    try:
        return joblib.load('hgb_maintenance_model.joblib')
    except:
        return None

model = load_model()

# ==========================================
# 4. MAIN INTERFACE HEADER
# ==========================================
st.markdown('<p class="main-title">Predictive Maintenance & Diagnostic System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Rotating Machinery Vibration Analysis Engine (HistGradientBoosting)</p>', unsafe_allow_html=True)

analysis_mode = st.radio(
    "Select Data Input Method:",
    ["Manual Entry", "Excel / CSV File Upload"],
    horizontal=True
)

st.markdown("---")

# ==========================================
# 5. MODE A: MANUAL ENTRY
# ==========================================
if analysis_mode == "Manual Entry":
    st.markdown('<p class="section-header">Asset Operational Context</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        selected_point_name = st.selectbox("Measurement Point Description (MptDesc)", list(measurement_points_mapping.keys()))
        mpt_desc = measurement_points_mapping[selected_point_name]
    with c2:
        rpm_val = st.number_input("Rotation Speed (RPM)", min_value=0.0, value=1500.0)

    st.markdown('<p class="section-header">Vibration Spectrum Magnitudes (mm/s RMS)</p>', unsafe_allow_html=True)
    cols = st.columns(5)
    harmonic_inputs = []

    for i, label in enumerate(harmonics_columns):
        with cols[i % 5]:
            # MODIFIÉ : format="%.10f" pour autoriser et afficher 10 chiffres après la virgule
            val = st.number_input(label, min_value=0.0, value=0.0, format="%.10f", step=0.0001, key=label)
            harmonic_inputs.append(val)

    st.markdown("---")
    if st.button("RUN SINGLE DIAGNOSTIC"):
        if model:
            features = [mpt_desc, rpm_val] + harmonic_inputs
            input_row_df = pd.DataFrame([features], columns=required_columns)
            
            pred = model.predict(input_row_df)[0]
            
            final_id = int(pred)
            diag = fault_names.get(final_id, "Normal Condition")
            
            st.markdown('<p class="section-header">Analysis Result</p>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="result-card">
                    <p style="color: #718096; text-transform: uppercase; letter-spacing: 1px; font-size: 12px;">Conclusion</p>
                    <h2 style="color: #1A365D; margin: 0;">{diag}</h2>
                    <p style="margin-top: 15px; color: #4A5568;">
                        The system identified patterns consistent with <b>{diag.lower()}</b> at <b>{selected_point_name}</b>.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Model file not found.")

# ==========================================
# 6. MODE B: FILE UPLOAD (EXCEL/CSV)
# ==========================================
else:
    st.markdown('<p class="section-header">Data Acquisition (File Import)</p>', unsafe_allow_html=True)
    st.info("Requirement: Upload an Excel or CSV file. The 'MptDesc' column can contain plain text names.")

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
                        results = []
                        
                        for i in range(len(df_input)):
                            row_data = df_input.iloc[i]
                            
                            text_mpt = str(row_data['MptDesc']).strip()
                            numeric_mpt = measurement_points_mapping.get(text_mpt, 1)
                            
                            features = [numeric_mpt, float(row_data['RPM'])] + [float(row_data[h]) for h in harmonics_columns]
                            input_row_df = pd.DataFrame([features], columns=required_columns)
                            
                            pred = model.predict(input_row_df)[0]
                            
                            final_id = int(pred)
                            diag = fault_names.get(final_id, "Normal Condition")
                            
                            results.append({
                                "Sample": i + 1,
                                "Measurement Point (MptDesc)": text_mpt,
                                "RPM": row_data['RPM'],
                                "Diagnostic Result": diag
                            })

                        st.markdown('<p class="section-header">Automated Diagnostic Report</p>', unsafe_allow_html=True)
                        
                        if len(results) == 1:
                            st.markdown(f"""
                                <div class="result-card">
                                    <p style="color: #718096; text-transform: uppercase; letter-spacing: 1px; font-size: 12px;">Conclusion</p>
                                    <h2 style="color: #1A365D; margin: 0;">{results[0]['Diagnostic Result']}</h2>
                                    <p style="margin-top: 15px; color: #4A5568;">
                                        The analysis identified <b>{results[0]['Diagnostic Result'].lower()}</b> for the location: <b>{results[0]['Measurement Point (MptDesc)']}</b>.
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.table(pd.DataFrame(results))
                    else:
                        st.error("Model file not found.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.markdown("""
            <div style="background-color: #FFFFFF; padding: 20px; border: 1px dashed #CBD5E0; color: #718096; text-align: center;">
                Please upload a dataset to begin the automated batch vibration analysis.
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# 7. FOOTER / SYSTEM INFO
# ==========================================
st.sidebar.markdown("---")
st.sidebar.caption("GIM Maintenance Hub - v3.4")
st.sidebar.caption("HistGradientBoosting Engine")
