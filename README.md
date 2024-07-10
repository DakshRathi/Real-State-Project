# Real Estate Project

## Overview
This project aims to provide various functionalities related to real estate analytics and property recommendation using data analysis and machine learning techniques.
Link to [Web-App](https://dakshrathi-real-state-project-web-app.streamlit.app/)

I have also deployed this project on AWS EC2 instance. [Link](https://drive.google.com/file/d/1QR1LXaWliwN5j-5PM5mzSEMA92WJR_D5/view?usp=sharing) to demonstration

## Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Modules](#modules)
  - [Price Predictor](#price-predictor)
  - [Analytics Module](#analytics-module)
  - [Recommender System](#recommender-system)
- [Dataset](#dataset)
- [Usage](#usage)

## Introduction
This repository contains a Streamlit-based application for real estate analytics and property recommendation. It leverages various data analysis techniques and machine learning models to provide insights into property prices, geographical distribution, and recommendations based on property features.

## Technologies Used
| Category             | Technologies                                     |
|----------------------|--------------------------------------------------|
| Programming Language | Python                                           |
| Version Control      | Git & GitHub                                     |
| Data Analysis        | Pandas, Numpy                                    |
| Visualization        | Matplotlib, Seaborn, Plotly                      |
| Machine Learning     | Scikit-Learn, XGBoost                            |
| Frontend & Backend   | Streamlit                                        |

## Project Structure
The project is organized into several modules, each serving a specific purpose related to real estate analytics and recommendation.

## Notebooks

### Jupyter Notebooks

1. **1-preprocessing.ipynb**
   - **Description**: Handles data preprocessing tasks including cleaning, normalization, and transformation.

2. **2-feature-engineering.ipynb**
   - **Description**: Implements feature engineering techniques such as creating new features and feature selection.
  
3. **3a-eda-univariate-analysis.ipynb**
   - **Description**: Conducts univariate analysis on various features to understand their distributions.

4. **3b-eda-multivariate-analysis.ipynb**
   - **Description**: Performs multivariate analysis to explore relationships between different features.

5. **4-outlier-treatment.ipynb**
   - **Description**: Implements outlier detection techniques and handles outlier treatment.

6. **5-missing-value-imputation.ipynb**
   - **Description**: Handles missing value imputation using appropriate strategies.

7. **6-feature-selection.ipynb**
   - **Description**: Implements feature selection techniques to choose the most relevant features for modeling.

8. **7a-baseline-model.ipynb**
   - **Description**: Builds baseline machine learning models for initial performance evaluation.

9. **7b-model-selection.ipynb**
   - **Description**: Selects the best machine learning model through comparative analysis and hyperparameter tuning.

10. **8-data-visualization-analytics-module.ipynb**
    - **Description**: Latitude and Longitude coordinates for sectors in Gurgaon were scraped using the `geopy.geocoders` library. After scraping the coordinates, a wordcloud CSV file (wordcloud.csv) was created to visualize sector features. This file is used in the analytics module for generating and displaying word clouds based on selected sectors.

11. **9-recommender-system.ipynb**
    - **Description**: Implements a recommender system for suggesting properties based on user preferences.

## Modules

### Price Predictor
- **Description**: Predicts property prices based on user inputs using a trained machine learning model.
- **Features**: User input collection, model prediction, and price range display.

### Analytics Module
- **Description**: Provides visual insights into property data such as price distribution, geographical mapping, and feature analysis.
- **Features**: Geo maps, scatter plots, pie charts, and distribution plots for detailed analysis.

### Recommender System
- **Description**: Recommends similar properties based on user-selected features using collaborative filtering techniques.
- **Features**: Property similarity calculation, search functionality, and recommendation display.

## Dataset
Disclaimer: I have scraped data from 99acres.com in this project and used those data for educational purpose only.

## Usage
To run the application locally:
1. Clone the repository: `git clone https://github.com/DakshRathi/Real-State-Project.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main application: `streamlit run home.py`
