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

    # Encodage
    data_encode = encoder.transform(input_data)

    # Scaling
    data_transform = scaler.transform(data_encode)

    # Prédiction
    prediction = model.predict(data_transform)

    st.success(f"💰 Prix estimé : {prediction[0]:,.0f} $")