import streamlit as st
import pickle
import numpy as np

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Personality App", layout="centered")

# ------------------ CUSTOM STYLE ------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }

    h1, h2, h3, h4 {
        color: #222222;
    }

    p, label {
        color: #333333 !important;
        font-size: 16px;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 700px;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    .card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))

# ------------------ HEADER ------------------
st.title("🧠 Personality Prediction")
st.markdown("### Discover whether you are an Introvert or Extrovert")

# ------------------ FORM CARD ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Enter Your Details")

time_alone = st.slider("Time spent alone (hrs)", 0, 24, 2)
social_events = st.slider("Social event attendance", 0, 10, 1)
going_out = st.slider("Outdoor activity frequency", 0, 10, 1)
friends = st.slider("Friends circle size", 0, 50, 5)
posts = st.slider("Social media post frequency", 0, 10, 1)

stage_fear = st.radio("Do you have stage fear?", ["Yes", "No"])
drained = st.radio("Do you feel drained after socializing?", ["Yes", "No"])

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ PREDICTION ------------------
if st.button("Predict Personality"):

    stage_fear_val = 1 if stage_fear == "Yes" else 0
    drained_val = 1 if drained == "Yes" else 0

    input_data = np.array([[time_alone, stage_fear_val, social_events,
                            going_out, drained_val, friends, posts]])

    prediction = model.predict(input_data)

    st.markdown("## Result")

    if prediction[0] == 1:
        st.success("🟢 You are an Extrovert")
    else:
        st.error("🔵 You are an Introvert")

    st.info("Prediction powered by Logistic Regression")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("Built with Streamlit | Clean UI Version")
