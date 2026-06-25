import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Prédiction de prix")
st.header("Prédiction des prix de maison en californie")
st.subheader("Dataset Housing.csv")
st.write("Ce programme est un modèle qui prédit les  prix des maisons en californie aux Etats unis.")

# Inputs
col1, col2 = st.columns(2)

with col1:
    longitude = st.slider("Longitude", -124.350000, -114.310000, -120.0)
    latitude = st.slider("Latitude", 32.540000, 41.950000, 37.0)
    housing_median_age = st.slider("Âge médian des logements",1,53,3)
    total_rooms = st.slider("Pièces", 2, 5699, 200)
    total_bedrooms = st.slider("Nombre de chambres",1,1190)

with col2:
    population = st.slider("Population", 3, 3500, 300)
    households = st.slider("Ménages",1,1150)
    median_income = st.slider("Revenu médian",0.499900,8.18,5.45)

ocean_proximity = st.selectbox(
    "Proximité de l'océan",
    ["INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
)

# Button
if st.button("🔮 Prédire le prix"):

    input_data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    # Feature engineering
    input_data["rooms_per_household"] = input_data["total_rooms"] / (input_data["households"] + 1e-5)
    input_data["bedrooms_per_room"] = input_data["total_bedrooms"] / (input_data["total_rooms"] + 1e-5)
    input_data["population_per_household"] = input_data["population"] / (input_data["households"] + 1e-5)

    input_data["ocean_proximity"] = input_data["ocean_proximity"].fillna("INLAND")

    #encoder

    cat_encoded = encoder.transform(input_data[["ocean_proximity"]])

    cat_encoded = pd.DataFrame(
        cat_encoded,
        columns=encoder.get_feature_names_out(["ocean_proximity"])
    )

    input_data = input_data.drop("ocean_proximity", axis=1)

    input_data = pd.concat(
        [input_data.reset_index(drop=True), cat_encoded],
        axis=1
    )

    # Scale
    X_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(X_scaled)

    st.success(f" Prix estimé : {prediction[0]:,.0f} $")
