import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st

st.title("Real Estate Price Predictor")
st.write("""
This app predicts the price that a house on the market will sell for,
given inputs such as property tax, insurance, number of beds, number of bathrooms, etc.
""")



# Prepare the form
with st.form("user_inputs"):
    st.subheader("House Listing Details")
    
    # year sold
    Year_Sold = st.number_input("Year Sold", 
                                  min_value=1900, 
                                  step=1)
    
    # property tax
    Property_Tax = st.number_input("Property Tax", 
                                  min_value=0, 
                                  step=1)
    
    # insurance
    Insurance = st.number_input("Insurance", 
                                  min_value=0, 
                                  step=1)
    # beds
    Beds = st.number_input("Number of Bedrooms", 
                                  min_value=0,
                                  max_value=10, 
                                  step=1)
    # baths
    Baths = st.number_input("Number of Bathrooms", 
                                  min_value=0,
                                  max_value=10,
                                  step=1)
    
    #sqft
    SqFt = st.number_input("Area (sq.ft)", 
                                  min_value=0, 
                                  step=1)
    # year_built
    Year_Built = st.number_input("Year Built", 
                                  min_value=1800, 
                                  step=1)
    
    # lot_size 
    Lot_Size = st.number_input("Lot Size", 
                                  min_value=0, 
                                  step=1)
    # basement
    Basement = st.selectbox("Does the house have a basement?", options=["Yes", "No"])
   
    # property_type_Bunglow
    Bungalow = st.selectbox("Is this house a bungalow (if not, it will be counted as a condo)?", options=["Yes", "No"])
    # property_type_Condo
    
    # Submit button
    submitted = st.form_submit_button("Predict House Price")


# Handle the dummy variables to pass to the model
if submitted:
    Has_Basement = 1 if Basement == "Yes" else 0
    Is_Bungalow = 1 if Bungalow == "Yes" else 0
    Is_Condo = 1 if Bungalow == "No" else 0
    
    # Convert Loan Amount Term and Credit History to integers
    Year_Sold = int(Year_Sold)
    Property_Tax = int(Property_Tax)
    Insurance = int(Insurance)
    Beds = int(Beds)
    Baths = int(Baths)
    SqFt = int(SqFt)
    Year_Built = int(Year_Built)
    Lot_Size = int(Lot_Size)
    
    # Calculated values (based on other values)
    Is_Popular = 1 if Beds == 2 and Baths == 2 else 0
    In_Recession = 1 if Year_Sold >= 2010 and Year_Sold <=2013 else 0
    Property_Age = Year_Sold - Year_Built

    # Prepare the input for prediction. This has to go in the same order as it was trained
    prediction_input = [[Year_Sold, Property_Tax, Insurance,
        Beds, Baths, SqFt, Year_Built,
        Lot_Size, Has_Basement, Is_Popular, In_Recession,
        Property_Age, Is_Bungalow, Is_Condo]]
    
    # Load the pickle model
    re_pickle = open(r"models/dt_RE_Model.pkl", "rb")
    re_model = pickle.load(re_pickle)
    re_pickle.close()

    # Make prediction
    new_prediction = re_model.predict(prediction_input)

    # Display result
    st.subheader("Prediction Result:")
    st.write(f"The predicted price is ${round(new_prediction[0],2)}.")

    st.subheader("Predicted vs. Actual Prices:")
    st.write("This is a graph of the predicted prices compared to the actual prices from the test dataset:")
    st.write("Note that this is a decision tree model, so the prediction can be one of a few numbers")
    st.image(r"photos/DTree_PredictvsActual.png")
