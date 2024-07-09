import streamlit as st
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, List

# Load the data and model pipeline
def load_data_and_model() -> Tuple[pd.DataFrame, any]:
    """
    Loads the preprocessed data and the trained model pipeline from pickle files.
    
    Returns:
    - df (pd.DataFrame): The preprocessed data frame.
    - pipeline (any): The trained model pipeline.
    """
    with open('Pickle-File/df.pkl','rb') as file:
        df = pickle.load(file)
    with open('Pickle-File/pipeline.pkl','rb') as file:
        pipeline = pickle.load(file)
    return df, pipeline

# Function to collect user inputs
def collect_user_inputs(df: pd.DataFrame) -> Tuple[str, str, float, float, str, str, float, float, float, str, str, str]:
    """
    Collects user inputs via Streamlit interface for property price prediction.

    Args:
    - df (pd.DataFrame): The preprocessed data frame containing necessary columns.

    Returns:
    - Tuple containing user inputs:
      (property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category)
    """
    st.header('Enter your inputs')
    col1, col2 = st.columns(2)
    with col1:
        property_type = st.selectbox('Property Type',['flat','house'])
        sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))
        bedrooms = float(st.selectbox('Number of Bedroom',sorted(df['bedRoom'].unique().tolist())))
        bathroom = float(st.selectbox('Number of Bathrooms',sorted(df['bathroom'].unique().tolist())))
        balcony = st.selectbox('Balconies',sorted(df['balcony'].unique().tolist()))
        property_age = st.selectbox('Property Age',sorted(df['agePossession'].unique().tolist()))

    with col2:
        built_up_area = float(st.number_input('Built Up Area'))
        servant_room = float(st.selectbox('Servant Room',[0.0, 1.0]))
        store_room = float(st.selectbox('Store Room',[0.0, 1.0]))
        furnishing_type = st.selectbox('Furnishing Type',sorted(df['furnishing_type'].unique().tolist()))
        luxury_category = st.selectbox('Luxury Category',sorted(df['luxury_category'].unique().tolist()))
        floor_category = st.selectbox('Floor Category',sorted(df['floor_category'].unique().tolist()))

    return property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category

# Function to predict price and display results
def predict_and_display_price(df: pd.DataFrame, pipeline: any, inputs: Tuple[str, str, float, float, str, str, float, float, float, str, str, str]) -> None:
    """
    Predicts the price based on user inputs using a pre-trained model pipeline and displays the predicted price range.

    Args:
    - df (pd.DataFrame): The preprocessed data frame containing necessary columns.
    - pipeline (any): The trained model pipeline for prediction.
    - inputs (Tuple): Tuple containing user inputs for prediction.

    Returns:
    - None
    """
    property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category = inputs
    
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony', 'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']

    one_df = pd.DataFrame(data, columns=columns)

    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = round(base_price - 0.22, 2)
    high = round(base_price + 0.22, 2)

    st.text("The price of the flat is between {} Cr and {} Cr".format(low,high))

# Main function to orchestrate the app
def main() -> None:
    """
    Main function to run the Streamlit application for price prediction based on user inputs.
    """
    st.set_page_config(page_title="Price Predictor", page_icon=":money_with_wings:")
    st.title('Price Predictor :money_with_wings:')
    df, pipeline = load_data_and_model()
    inputs = collect_user_inputs(df)
    
    if st.button('Predict'):
        predict_and_display_price(df, pipeline, inputs)

# Execute the main function
if __name__ == '__main__':
    main()