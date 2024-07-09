import streamlit as st

# Set page title and icon
st.set_page_config(
    page_title="Gurgaon Real Estate Analytics App",
    page_icon=":house_buildings:",
)

# Main title and introduction
st.write("# Welcome to Gurgaon Real Estate Analytics App :house_with_garden:")
st.write("Explore various modules to analyze and predict real estate trends in Gurgaon.")

# Sidebar with module selection
st.sidebar.title("Modules")

st.write("Welcome to the home page of Gurgaon Real Estate Analytics App.")
st.write("Select a module from the sidebar to explore more.")

st.write("### :money_with_wings: Price Predictor Module")
st.write("Predict property prices based on various factors such as property type, sector, bedrooms, bathrooms, etc.")
st.write("Use advanced machine learning models to estimate price ranges.")
col1, col2 = st.columns(2)
with col1:
    st.write("### 🔍 Analytics Module")
    st.write("Explore comprehensive analytics including:")
    st.write("- Geographic distribution of property prices.")
    st.write("- Word cloud analysis of property features.")
    st.write("- Scatter plots and distribution plots for price analysis.")

with col2:
    st.write("### 📍 Apartment Recommender Module")
    st.write("Find similar apartments based on:")
    st.write("- Location proximity.")
    st.write("- Similarity in property features.")
    st.write("Receive personalized recommendations for apartment hunting.")