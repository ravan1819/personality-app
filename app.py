import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Personality App", layout="centered")

model = pickle.load(open("model.pkl", "rb"))

# HEADER
st.title("🧠 Personality Prediction App")
st.markdown("### Discover your personality type")

st.divider()

# INPUTS
st.subheader("Enter Your Details")

time_alone = st.slider("Time spent alone (hrs)", 0, 24, 2)
social_events = st.slider("Social event attendance", 0, 10, 1)
going_out = st.slider("Outdoor activity frequency", 0, 10, 1)
friends = st.slider("Friends circle size", 0, 50, 5)
posts = st.slider("Social media post frequency", 0, 10, 1)

stage_fear = st.radio("Do you have stage fear?", ["Yes", "No"])
drained = st.radio("Do you feel drained after socializing?", ["Yes", "No"])

st.divider()

# PREDICT
if st.button("Predict Personality"):

    stage_fear = 1 if stage_fear == "Yes" else 0
    drained = 1 if drained == "Yes" else 0

    input_data = np.array([[time_alone, stage_fear, social_events,
                            going_out, drained, friends, posts]])

    prediction = model.predict(input_data)

    st.subheader("Result")

    if prediction[0] == 1:
        st.success("🟢 You are an Extrovert")
    else:
        st.info("🔵 You are an Introvert")
