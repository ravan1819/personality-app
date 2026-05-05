import streamlit as st
import pickle
import numpy as np

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Personality App", layout="centered")

# ------------------ CUSTOM STYLE ------------------
st.markdown("""
    <style>
    body {
        background-color: white;
    }
    .stApp {
        background-color: white;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))

# ------------------ HEADER ------------------
st.title("🧠 Personality Prediction App")
st.markdown("### Find out if you're an Introvert or Extrovert")

st.markdown("---")

# ------------------ INPUT SECTION ------------------
st.subheader("📋 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    time_alone = st.slider("Time spent alone (hrs)", 0, 24, 2)
    social_events = st.slider("Social event attendance", 0, 10, 1)
    friends = st.slider("Friends circle size", 0, 50, 5)

with col2:
    going_out = st.slider("Outdoor activity frequency", 0, 10, 1)
    posts = st.slider("Social media post frequency", 0, 10, 1)

stage_fear = st.radio("Do you have stage fear?", ["Yes", "No"])
drained = st.radio("Do you feel drained after socializing?", ["Yes", "No"])

st.markdown("---")

# ------------------ PREDICTION ------------------
if st.button("🔍 Predict Personality"):

    stage_fear_val = 1 if stage_fear == "Yes" else 0
    drained_val = 1 if drained == "Yes" else 0

    input_data = np.array([[time_alone, stage_fear_val, social_events,
                            going_out, drained_val, friends, posts]])

    prediction = model.predict(input_data)

    st.markdown("## 🎯 Result")

    if prediction[0] == 1:
        st.success("🟢 You are an **Extrovert**")
    else:
        st.error("🔵 You are an **Introvert**")

    st.info("This prediction is based on Logistic Regression trained on behavioral data.")

st.markdown("---")

# ------------------ FOOTER ------------------
st.markdown("💡 Built with Streamlit | Machine Learning Project")
