import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
import os

st.set_page_config(page_title="Autism Prediction", page_icon="🧩", layout="wide")

# --- Styles (simple, clean and slightly modern) ---
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(180deg, #ffffff 0%, #f2f9ff 100%);
    }
    .stApp > header {display: none}
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 24px;
        border: none;
        box-shadow: 0 8px 30px rgba(14, 42, 80, 0.06);
        max-width: 1100px;
        margin: 16px auto;
    }
    .big-title {font-family: 'Segoe UI', Roboto, sans-serif; font-size:34px; font-weight:800; color:#00bfff;}
    .muted {color:#475569}
    .small {font-size:13px}
    </style>
    """,
    unsafe_allow_html=True,
)

# Top header
with st.container():
    c1, _ = st.columns([3, 1])
    with c1:
        st.markdown('<div class="big-title">Autism Prediction</div>', unsafe_allow_html=True)


# Load model and encoders
@st.cache_data(show_spinner=False)
def load_artifacts():
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Could not load best_model.pkl: {e}")
        model = None

    try:
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
    except Exception as e:
        st.error(f"Could not load encoders.pkl: {e}")
        encoders = {}

    return model, encoders

model, encoders = load_artifacts()

# Determine a sensible default for the numeric 'result' feature.
# Try to compute median from available dataset (final_data.xls) if present, otherwise default to 0.0
default_result = 0.0
try:
    data_path = "final_data.xls"
    if os.path.exists(data_path):
        df_full = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_excel(data_path)
        if 'result' in df_full.columns:
            default_result = float(df_full['result'].median())
except Exception:
    default_result = 0.0

FEATURE_ORDER = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'ethnicity', 'jaundice', 'austim',
    'contry_of_res', 'used_app_before', 'result', 'relation'
]

# Utility: preprocess dataframe by applying encoders where needed
def preprocess_input(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    # Ensure all expected columns exist
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Cast numeric columns
    for num_col in ['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score','age','result']:
        df[num_col] = pd.to_numeric(df[num_col], errors='coerce')

    # Apply encoders
    for col, enc in encoders.items():
        if col in df.columns:
            # If values not in classes_, this will raise; handle gracefully
            try:
                df[col] = enc.transform(df[col])
            except Exception:
                # Try to align by mapping unseen categories to a nearest known option when possible
                df[col] = df[col].astype(str)
                mapping = {v: i for i, v in enumerate(enc.classes_.astype(str))}
                df[col] = df[col].map(mapping)
                if df[col].isnull().any():
                    raise ValueError(f"Column '{col}' contains categories not seen during training. Allowed: {list(enc.classes_)}")
    return df[FEATURE_ORDER]

# Sidebar: mode selection
mode = st.sidebar.selectbox("Select mode", ["Single input (one person)", "Batch file (CSV/Excel)"])

if mode.startswith("Single"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # Layout inputs in columns for a compact form
    with st.form(key="single_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            A1 = st.number_input('A1_Score', min_value=0, max_value=10, value=1)
            A2 = st.number_input('A2_Score', min_value=0, max_value=10, value=1)
            A3 = st.number_input('A3_Score', min_value=0, max_value=10, value=1)
            A4 = st.number_input('A4_Score', min_value=0, max_value=10, value=1)
            A5 = st.number_input('A5_Score', min_value=0, max_value=10, value=1)
        with col2:
            A6 = st.number_input('A6_Score', min_value=0, max_value=10, value=1)
            A7 = st.number_input('A7_Score', min_value=0, max_value=10, value=1)
            A8 = st.number_input('A8_Score', min_value=0, max_value=10, value=1)
            A9 = st.number_input('A9_Score', min_value=0, max_value=10, value=1)
            A10 = st.number_input('A10_Score', min_value=0, max_value=10, value=1)
        with col3:
            age = st.number_input('age', min_value=1, max_value=120, value=25)
            # Populate categorical options from encoders if available
            def options_for(col, default=[]):
                return list(getattr(encoders.get(col, None), 'classes_', default))

            gender = st.selectbox('gender', options_for('gender', ['m','f']))
            ethnicity = st.selectbox('ethnicity', options_for('ethnicity', ['White-European']))
            jaundice = st.selectbox('jaundice', options_for('jaundice', ['no','yes']))
            contry_of_res = st.selectbox('contry_of_res', options_for('contry_of_res', ['United States']))
            used_app_before = st.selectbox('used_app_before', options_for('used_app_before', ['no','yes']))
            relation = st.selectbox('relation', options_for('relation', ['Self']))

        submit = st.form_submit_button('Predict')

    if submit:
        # Build DataFrame
        input_dict = {
            'A1_Score': A1, 'A2_Score': A2, 'A3_Score': A3, 'A4_Score': A4, 'A5_Score': A5,
            'A6_Score': A6, 'A7_Score': A7, 'A8_Score': A8, 'A9_Score': A9, 'A10_Score': A10,
            'age': age, 'gender': gender, 'ethnicity': ethnicity, 'jaundice': jaundice,
            'contry_of_res': contry_of_res, 'used_app_before': used_app_before, 'relation': relation
        }
        input_df = pd.DataFrame([input_dict])
        # Fill hidden 'austim' column with a sensible default from training encoders (or 'no')
        default_austim = getattr(encoders.get('austim', None), 'classes_', ['no'])[0]
        input_df['austim'] = default_austim
        # Fill hidden numeric 'result' with training median-derived default so the model has a value
        input_df['result'] = default_result
        try:
            X = preprocess_input(input_df, encoders)
            pred = model.predict(X)[0]
            proba = None
            prob_autism = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                # determine which column corresponds to class '1' (autism)
                try:
                    idx = list(model.classes_).index(1)
                except Exception:
                    idx = 1 if len(proba) > 1 else 0
                prob_autism = proba[idx] * 100

            # Clear, user-focused result message (probability of autism)
            if prob_autism is not None:
                st.markdown(f"### You have **{prob_autism:.2f}%** chance of autism")
            else:
                # fallback to predicted class label
                if pred == 1:
                    st.markdown("### Prediction: Positive for Autism Traits")
                else:
                    st.markdown("### Prediction: Negative for Autism Traits")

            if prob_autism is not None:
                st.write(f"Model probability (autism): {prob_autism:.2f}%")

        except Exception as e:
            st.error(f"Could not make prediction: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload file", type=["csv","xls","xlsx"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            fname = uploaded.name.lower()
            if fname.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            st.dataframe(df.head())

            if st.button('Run batch prediction'):
                try:
                    # Fill missing hidden 'austim' with default from encoders (or 'no')
                    if 'austim' not in df.columns:
                        df['austim'] = getattr(encoders.get('austim', None), 'classes_', ['no'])[0]
                    # Fill missing numeric 'result' with training median default
                    if 'result' not in df.columns:
                        df['result'] = default_result
                    X = preprocess_input(df, encoders)
                    preds = model.predict(X)
                    df_out = df.copy()
                    # Human-readable predicted label
                    df_out['predicted_autism'] = ['Yes' if int(p)==1 else 'No' for p in preds]

                    # If model supports probabilities, add autism probability column
                    if hasattr(model, 'predict_proba'):
                        probas = model.predict_proba(X)
                        try:
                            idx = list(model.classes_).index(1)
                        except Exception:
                            idx = 1 if probas.shape[1] > 1 else 0
                        df_out['autism_probability'] = (probas[:, idx] * 100).round(2)

                    st.success('Predictions complete')
                    st.dataframe(df_out.head())

                    # Provide download
                    towrite = BytesIO()
                    df_out.to_csv(towrite, index=False)
                    towrite.seek(0)
                    st.download_button(label='Download results as CSV', data=towrite, file_name='predictions.csv', mime='text/csv')
                except Exception as e:
                    st.error(f'Prediction failed: {e}')

        except Exception as e:
            st.error(f'Could not read uploaded file: {e}')

    st.markdown('</div>', unsafe_allow_html=True)

