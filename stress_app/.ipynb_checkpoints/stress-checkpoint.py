import streamlit as st
import numpy as np
import pickle

st.title("Prediksi Stress Level")

# load model
beta = pickle.load(open("model_regresi.pkl","rb"))

# INPUT DENGAN BATAS
productivity = st.number_input(
    "Work Productivity Score (1-10)",
    min_value=1.0,
    max_value=10.0,
    step=1.0
)

sleep = st.number_input(
    "Sleep Hours (0-12 jam)",
    min_value=0.0,
    max_value=12.0,
    step=0.5
)

weekend = st.number_input(
    "Weekend Screen Time Hours (0-24 jam)",
    min_value=0.0,
    max_value=24.0,
    step=0.5
)

caffeine = st.number_input(
    "Caffeine Intake Cups (0-10)",
    min_value=0,
    max_value=10,
    step=1
)

# PREDIKSI
if st.button("Prediksi"):

    X = np.array([1, productivity, sleep, weekend, caffeine])

    pred = (X @ beta).item()

    # kategori stress
    if pred < 2:
        kategori = "Sangat Rendah"
    elif pred < 4:
        kategori = "Rendah"
    elif pred < 6:
        kategori = "Sedang"
    elif pred < 8:
        kategori = "Tinggi"
    else:
        kategori = "Sangat Tinggi"

    st.success(f"Prediksi Stress Level: {pred:.2f}")
    st.write("Kategori Stress:", kategori)