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

st.title("🌾 Smart Agriculture Assistant")

# === Session state for control ===
if 'input_done' not in st.session_state:
    st.session_state.input_done = False
if 'show_charts' not in st.session_state:
    st.session_state.show_charts = False
if 'user_df' not in st.session_state:
    st.session_state.user_df = pd.DataFrame()

# === Input Form ===
if not st.session_state.input_done:
    st.header("🔢 Enter Soil and Weather Parameters")
    user_input = {}
    error_flag = False
    for col in feature_names:
        max_val = feature_max_values.get(col, 1e9)
        val = st.number_input(f"{col} (Max: {max_val})", value=0.0, key=col)
        if val > max_val:
            st.error(f"❌ Invalid input for **{col}**. Maximum allowed is {max_val}.")
            error_flag = True
        user_input[col] = val

    user_df = pd.DataFrame([user_input])

    if st.button("🚀 Predict Crop, Yield & Fertilizer"):
        if error_flag:
            st.warning("❌ Please correct the invalid input(s) before proceeding.")
        else:
            st.session_state.user_df = user_df
            st.session_state.input_done = True
            st.rerun()

else:
    user_df = st.session_state.user_df

    # 🚫 Reject all-zero input
    if (user_df == 0).any(axis=1).bool():
        st.error("❌ Please fill in all fields. No input should be zero.")
    else:
        # 🔄 Scale input
        user_scaled = scaler.transform(user_df)

        # 🔍 Predictions
        pred_crop = le_crop.inverse_transform(crop_model.predict(user_scaled))[0]
        pred_yield = yield_model.predict(user_scaled)[0]
        pred_fert = le_fert.inverse_transform(fert_model.predict(user_scaled))[0]

        st.success(f"🧾 **Recommended Crop**: {pred_crop}")
        st.info(f"📈 **Predicted Yield**: {pred_yield:.2f} quintals/hectare")
        st.warning(f"💡 **Recommended Fertilizer**: {pred_fert}")

    if st.button("🔁 Enter New Values"):
        st.session_state.input_done = False
        st.rerun()
