import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Title of the app
st.title("Support Vector Machine(SVM)")

# Subtitle
st.subheader("Practice using SVM for regression with California Housing Prediction and classification tasks with Iris Dataset.")

# Sidebar for app navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the mode", ["Regression with California Housing Prediction", "Classification with Iris Dataset"])

@st.dialog("Predict species for Iris dataset")
def clf_predict():
    # Display input parameters
    st.write("Input Parameters:")
    st.write(f"Sepal Length (cm): {SepalLengthCm}")
    st.write(f"Sepal Width (cm): {SepalWidthCm}")
    st.write(f"Petal Length (cm): {PetalLengthCm}")
    st.write(f"Petal Width (cm): {PetalWidthCm}")

    # Prepare the input data
    input_data = np.array([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])

    # Load the trained model and scaler
    with open('model/clf/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/clf/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Predict the species
    prediction = model.predict(scaled_input)
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    predicted_species = species_map[prediction[0]]

    st.write(f"Predicted Iris Species: **{predicted_species}**")

@st.dialog("Predict housing price in california")
def reg_predict():
    # Display input parameters
    st.write("Input Parameters:")
    st.write(f"Median Income: {MedInc}")
    st.write(f"Median House Age: {HouseAge}")
    st.write(f"Average Rooms per Household: {AveRooms}")
    st.write(f"Average Bedrooms per Household: {AveBedrms}")
    st.write(f"Population: {Population}")
    st.write(f"Average Occupants per Household: {AveOccup}")
    st.write(f"Latitude: {Latitude}")
    st.write(f"Longitude: {Longitude}")

    # Prepare the input data
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup]])

    
    

    # Load the trained model from the pickle file
    with open('model/reg/model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Load the trained model from the pickle file
    with open('model/reg/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Scale the input data using the loaded StandardScaler
    scaled_input = scaler.transform(input_data)

    # Use the loaded model to predict the FWI
    prediction = model.predict(scaled_input).astype(float)* 100000
    st.write(f"Predicted Housing Price: ${prediction[0]:.2f}")


if app_mode == "Regression with California Housing Prediction":
    # California Housing Prediction
    st.header("Predict California Housing Prices")

    # Feature Inputs for California Housing
    MedInc = st.number_input('Median Income (in tens of thousands)', min_value=0.0, value=3.0, step=0.1)
    HouseAge = st.number_input('Median House Age', min_value=0, max_value=100, value=25, step=1)
    AveRooms = st.number_input('Average Rooms per Household', min_value=0.0, value=5.0, step=0.1)
    AveBedrms = st.number_input('Average Bedrooms per Household', min_value=0.0, value=1.0, step=0.1)
    Population = st.number_input('Population', min_value=0, value=500, step=1)
    AveOccup = st.number_input('Average Occupants per Household', min_value=0.0, value=3.0, step=0.1)
    Latitude = st.number_input('Latitude', min_value=32.0, max_value=42.0, value=35.0, step=0.1)
    Longitude = st.number_input('Longitude', min_value=-125.0, max_value=-114.0, value=-120.0, step=0.1)

    # Check if any input is missing
    inputs_filled = all([
        MedInc is not None,
        HouseAge is not None,
        AveRooms is not None,
        AveBedrms is not None,
        Population is not None,
        AveOccup is not None,
        Latitude is not None,
        Longitude is not None
    ])

    # Button to submit the form
    if st.button('Predict Housing Price.', disabled=not inputs_filled):
        reg_predict()

    # Display a message if the button is inactive
    if not inputs_filled:
        st.write("Please fill out all the inputs to enable the prediction.")


elif app_mode == "Classification with Iris Dataset":
    # Load Iris Dataset
    # Iris Dataset Classification
    st.header("Predict Iris Species")

    # Feature Inputs for Iris Dataset
    SepalLengthCm = st.number_input('Sepal Length (cm)', min_value=0.0, value=5.0, step=0.1)
    SepalWidthCm = st.number_input('Sepal Width (cm)', min_value=0.0, value=3.0, step=0.1)
    PetalLengthCm = st.number_input('Petal Length (cm)', min_value=0.0, value=1.5, step=0.1)
    PetalWidthCm = st.number_input('Petal Width (cm)', min_value=0.0, value=0.2, step=0.1)

    # Check if any input is missing
    inputs_filled = all([
        SepalLengthCm is not None,
        SepalWidthCm is not None,
        PetalLengthCm is not None,
        PetalWidthCm is not None
    ])
        # Button to submit the form
    if st.button('Predict Iris Species', disabled=not inputs_filled):
        clf_predict()

    # Display a message if the button is inactive
    if not inputs_filled:
        st.write("Please fill out all the inputs to enable the prediction.")

    