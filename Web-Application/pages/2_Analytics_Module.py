import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Function to load data
def load_data() -> tuple:
    """
    Loads and returns the necessary dataframes.

    Returns:
        tuple: A tuple containing two dataframes: new_df (properties-lat-long.csv) and df_wordcloud (wordcloud.csv).
    """
    new_df = pd.read_csv("Pickle-File/properties-lat-long.csv")
    df_wordcloud = pd.read_csv("Pickle-File/wordcloud.csv")
    return new_df, df_wordcloud

# Function to display Geo map
def display_geo_map(data: pd.DataFrame) -> None:
    """
    Displays a scatter map of sector price per sqft.

    Args:
        data (pd.DataFrame): DataFrame containing sector-wise data.
    """
    st.header('🗺️ Sector Price per Sqft Geomap')
    data = data[['sector', 'price', 'price_per_sqft', 'built_up_area', 'Latitude', 'Longitude']]
    group_df = data.groupby('sector').mean()[['price', 'price_per_sqft', 'built_up_area', 'Latitude', 'Longitude']]
    
    fig = px.scatter_mapbox(group_df, lat="Latitude", lon="Longitude", color="price_per_sqft", size='built_up_area',
                            color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
                            mapbox_style="open-street-map", width=1200, height=700, hover_name=group_df.index)
    
    st.plotly_chart(fig, use_container_width=True)

# Function to generate and display word cloud
def generate_word_cloud(df_wordcloud: pd.DataFrame) -> None:
    """
    Generates and displays a word cloud based on selected sector or overall.

    Args:
        df_wordcloud (pd.DataFrame): DataFrame containing sector-wise word cloud data.
    """
    st.header('☁️ Sector Features Wordcloud')

    sector_options = df_wordcloud['sector'].unique().tolist()
    sector_options.insert(0, 'overall')
    selected_sector = st.selectbox('Select Sector:', sector_options)

    if selected_sector == 'overall':
        feature_text = ' '.join(df_wordcloud['features'].dropna().tolist())
    else:
        feature_text = df_wordcloud[df_wordcloud['sector'] == selected_sector]['features'].values[0]

    if st.button('Generate Word Cloud'):
        wordcloud = WordCloud(width=800, height=800,
                              background_color='black',
                              stopwords=set(['s']),  # Any stopwords you'd like to exclude
                              min_font_size=10).generate(feature_text)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig)


# Function to display Area Vs Price scatter plot
def display_area_price(new_df: pd.DataFrame) -> None:
    """
    Displays a scatter plot of area versus price based on selected property type.

    Args:
        new_df (pd.DataFrame): DataFrame containing property data.
    """
    st.header('🏘️ Area Vs Price')

    property_type = st.selectbox('Select Property Type', ['flat', 'house'])

    filtered_df = new_df[new_df['property_type'] == property_type]
    fig = px.scatter(filtered_df, x="built_up_area", y="price", color="bedRoom")

    st.plotly_chart(fig, use_container_width=True)

# Function to display BHK Pie Chart
def display_bhk_pie_chart(new_df: pd.DataFrame) -> None:
    """
    Displays a pie chart of BHK distribution based on selected sector or overall.

    Args:
        new_df (pd.DataFrame): DataFrame containing property data.
    """
    st.header('🍰 BHK Pie Chart')

    sector_options = new_df['sector'].unique().tolist()
    sector_options.insert(0, 'overall')
    selected_sector = st.selectbox('Select Sector', sector_options)

    if selected_sector == 'overall':
        fig = px.pie(new_df, names='bedRoom')
    else:
        fig = px.pie(new_df[new_df['sector'] == selected_sector], names='bedRoom')

    st.plotly_chart(fig, use_container_width=True)

# Function to display Side by Side BHK price comparison
def display_bhk_price_comparison(new_df: pd.DataFrame) -> None:
    """
    Displays a box plot comparing BHK prices.

    Args:
        new_df (pd.DataFrame): DataFrame containing property data.
    """
    st.header('📊 Side by Side BHK price comparison')

    fig = px.box(new_df[new_df['bedRoom'] <= 4], x='bedRoom', y='price', title='BHK Price Range')

    st.plotly_chart(fig, use_container_width=True)

# Function to display Side by Side Distplot for property type using Plotly
def display_distplot(new_df: pd.DataFrame) -> None:
    """
    Displays a distribution plot for house and flat prices using Plotly.

    Args:
        new_df (pd.DataFrame): DataFrame containing property data.
    """
    st.header('📈 Side by Side Distplot for property type')

    fig = px.histogram(new_df, x="price", color="property_type", nbins=100, opacity=0.7,
                       marginal="box",  # Include box plot
                       labels={"price": "Price (in crore)", "property_type": "Property Type"},
                       histnorm='probability density',  # Normalize to show probability density
                       title="Distribution of House and Flat Prices")

    st.plotly_chart(fig, use_container_width=True)

# Main Streamlit UI
def main() -> None:
    """
    Main function to orchestrate the Streamlit app for analytics module.
    """
    st.set_page_config(page_title="Analytics Module", page_icon="📊")
    st.title('🔍 Analytics Module')

    # Load data
    new_df, df_wordcloud = load_data()

    # Display Geo map
    display_geo_map(new_df)

    # Generate Word Cloud
    generate_word_cloud(df_wordcloud)

    # Display Area Vs Price
    display_area_price(new_df)

    # Display BHK Pie Chart
    display_bhk_pie_chart(new_df)

    # Display Side by Side BHK price comparison
    display_bhk_price_comparison(new_df)

    # Display Side by Side Distplot for property type
    display_distplot(new_df)

# Execute the main function
if __name__ == '__main__':
    main()