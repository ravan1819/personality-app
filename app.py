import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🧠 Personality Prediction App")

st.write("Enter details to predict personality")

# Inputs
time_alone = st.number_input("Time spent alone")
stage_fear = st.selectbox("Stage Fear", ["Yes", "No"])
social_events = st.number_input("Social event attendance")
going_out = st.number_input("Going outside")
drained = st.selectbox("Drained after socializing", ["Yes", "No"])
friends = st.number_input("Friends circle size")
posts = st.number_input("Post frequency")

# Convert categorical
stage_fear = 1 if stage_fear == "Yes" else 0
drained = 1 if drained == "Yes" else 0

# Predict
if st.button("Predict"):
    input_data = np.array([[time_alone, stage_fear, social_events, going_out, drained, friends, posts]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Extrovert")
    else:
        st.success("Introvert")