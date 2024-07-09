import streamlit as st
import pickle
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union

# Load data and models
st.set_page_config(page_title="Recommend Apartments",  page_icon='📍')

# Loading data
location_df: pd.DataFrame = pickle.load(open('Pickle-File/location_df.pkl', 'rb'))
cosine_sim1: np.ndarray = pickle.load(open('Pickle-File/cosine_sim1.pkl', 'rb'))
cosine_sim2: np.ndarray = pickle.load(open('Pickle-File/cosine_sim2.pkl', 'rb'))
cosine_sim3: np.ndarray = pickle.load(open('Pickle-File/cosine_sim3.pkl', 'rb'))

# Function to recommend properties based on similarity scores
def recommend_properties_with_scores(property_name: str, top_n: int = 5) -> pd.DataFrame:
    """
    Recommends properties similar to the given property based on pre-computed similarity scores.

    Args:
    - property_name (str): The name of the property to find similar properties for.
    - top_n (int): Number of top similar properties to recommend.

    Returns:
    - recommendations_df (pd.DataFrame): DataFrame containing recommended properties and their similarity scores.
    """
    cosine_sim_matrix: np.ndarray = 0.6 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3

    # Clean the property name to remove any extra spaces
    property_name = property_name.strip()

    # Get the similarity scores for the property using its name as the index
    sim_scores: List[Tuple[int, float]] = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))

    # Sort properties based on the similarity scores
    sorted_scores: List[Tuple[int, float]] = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices and scores of the top_n most similar properties
    top_indices: List[int] = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores: List[float] = [i[1] for i in sorted_scores[1:top_n + 1]]

    # Retrieve the names of the top properties using the indices
    top_properties: List[str] = location_df.index[top_indices].tolist()

    # Create a dataframe with the results
    recommendations_df: pd.DataFrame = pd.DataFrame({
        'Property Name': top_properties,
        'Similarity Score': top_scores
    })

    return recommendations_df

# Function to display search results
def display_search_results(selected_location: str, radius: float) -> None:
    """
    Displays search results based on selected location and radius.

    Args:
    - selected_location (str): The selected location to search around.
    - radius (float): The radius in kilometers to search within.
    """
    st.subheader('Search Results:')
    result_ser: pd.Series = location_df[location_df[selected_location] < radius * 1000][selected_location].sort_values()
    if result_ser.empty:
        st.text('No results to display.')
    else:
        for key, value in result_ser.items():
            st.text(f'{key} --> {round(value/1000)} kms')

# Function to handle the main Streamlit app
def main() -> None:
    """
    Main function to orchestrate the Streamlit app for recommending apartments.
    """
    st.title('Apartment Recommendation System 🏠')
    st.header('Select Location and Radius 📍')
    col1, col2 = st.columns([2, 3])
    
    with col1:
        selected_location: str = st.selectbox('Location', sorted(location_df.columns.to_list()))
        radius: float = st.number_input('Radius in Kms')
        search_clicked: bool = st.button('Search')
    
    with col2:
        if search_clicked:
            display_search_results(selected_location, radius)

    st.header('Recommend Apartments 🌟')
    selected_apartment: Optional[str] = st.selectbox('Select an apartment', sorted(location_df.index.to_list()))

    if st.button('Recommend'):
        if selected_apartment:
            recommendation_df: pd.DataFrame = recommend_properties_with_scores(selected_apartment)
            recommendation_df.index = np.arange(1, len(recommendation_df) + 1)
            st.dataframe(recommendation_df)
        else:
            st.warning('Please select an apartment to recommend.')

if __name__ == '__main__':
    main()