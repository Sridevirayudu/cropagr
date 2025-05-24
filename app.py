import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load models and encoders ===
crop_model = joblib.load('crop_model.pkl')
yield_model = joblib.load('yield_model.pkl')
fert_model = joblib.load('fert_model.pkl')
le_crop = joblib.load('le_crop.pkl')
le_fert = joblib.load('le_fert.pkl')
feature_names = joblib.load('feature_names.pkl')
scaler = joblib.load('scaler.pkl')
feature_max_values = joblib.load('feature_max_values.pkl')
feature_min_values = joblib.load('feature_min_values.pkl')

# === Friendly display names (only for UI) ===
friendly_labels = {
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "ph": "pH",
    "rainfall": "Rainfall (mm)"
}

st.title("🌾 Smart Agriculture Assistant")

# === Session state control ===
if 'input_done' not in st.session_state:
    st.session_state.input_done = False
if 'user_df' not in st.session_state:
    st.session_state.user_df = pd.DataFrame()
if 'predict_pressed' not in st.session_state:
    st.session_state.predict_pressed = False

# === Input Form ===
if not st.session_state.input_done:
    st.header("🔢 Enter Soil and Weather Parameters")

    user_input = {}
    error_flag = False

    for col in feature_names:
        label = friendly_labels.get(col, col)
        min_val = feature_min_values.get(col, 0.0)
        max_val = feature_max_values.get(col, 1e9)
        input_key = f"input_{col}"

        user_val = st.text_input(
            label=label,
            placeholder=f"Enter {label} (Min: {min_val}, Max: {max_val})",
            key=input_key
        )

        if st.session_state.predict_pressed:
            if user_val.strip() == "":
                st.error(f"❌ Please enter a value for **{label}**.")
                error_flag = True
                continue
            try:
                val = float(user_val)
                if val < min_val:
                    st.error(f"❌ **{label}** must be ≥ {min_val}.")
                    error_flag = True
                elif val > max_val:
                    st.error(f"❌ **{label}** must be ≤ {max_val}.")
                    error_flag = True
                user_input[col] = val
            except ValueError:
                st.error(f"❌ **{label}** must be a number.")
                error_flag = True
        else:
            user_input[col] = user_val if user_val.strip() else None

    # === Predict button logic ===
    if st.button("🚀 Predict Crop, Yield & Fertilizer"):
        st.session_state.predict_pressed = True

        validated_input = {}
        for col in feature_names:
            val_str = st.session_state.get(f"input_{col}", "").strip()
            min_val = feature_min_values.get(col, 0.0)
            max_val = feature_max_values.get(col, 1e9)
            try:
                val = float(val_str)
                if val < min_val or val > max_val:
                    st.warning(f"❌ {friendly_labels.get(col, col)} must be between {min_val} and {max_val}.")
                    error_flag = True
                validated_input[col] = val
            except:
                st.warning(f"❌ {friendly_labels.get(col, col)} must be a valid number.")
                error_flag = True

        if not error_flag and len(validated_input) == len(feature_names):
            user_df = pd.DataFrame([validated_input])
            st.session_state.user_df = user_df
            st.session_state.input_done = True
            st.session_state.predict_pressed = False
            st.rerun()
        else:
            st.warning("❌ Please correct all inputs before proceeding.")

# === Output Section ===
else:
    user_df = st.session_state.user_df

    if (user_df == 0).any(axis=1).bool():
        st.error("❌ Please fill in all fields. No input should be zero.")
    else:
        user_scaled = scaler.transform(user_df)

        pred_crop = le_crop.inverse_transform(crop_model.predict(user_scaled))[0]
        pred_yield = yield_model.predict(user_scaled)[0]
        pred_fert = le_fert.inverse_transform(fert_model.predict(user_scaled))[0]

        st.success(f"🧾 **Recommended Crop**: {pred_crop}")
        st.info(f"📈 **Predicted Yield**: {pred_yield:.2f} quintals/hectare")
        st.warning(f"💡 **Recommended Fertilizer**: {pred_fert}")

    if st.button("🔁 Enter New Values"):
        st.session_state.input_done = False
        st.session_state.predict_pressed = False
        st.rerun()
