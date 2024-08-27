# AI_Project_3_Group_01
A data analysis class project 3 for Columbia University's AI bootcamp.

## Title Placeholder

<div align='center'>
    <img src='' height='300' title='Placeholder' alt='Placeholder' />
## Leveraging Reviews to Understand Consumer Sentiments ##
*Footnote of image*[^1]
</div>

## Project Team Members:
* Ramona Ciobanu
* Christian Leon
* Leigh Nash
* Angelica Padilla
* Odele Pax
* Vanessa Wright

---

## Table of Contents

* [Project Details](#Project-Details)
* [Data](#Data)
* [Project Management](#Project-Management)
* [Findings](#Findings)
* [Citations and Licenses](#Citations-and-Licenses)

---

## Project Details

***Thesis:***

This project aims to explore the relationship between various business attributes and review ratings. We hypothesize that certain business attributes (e.g., reviews, location, service quality, etc.) can be used to train AI to analyze and make suggestions to improve the businesses.

---

## Data

* List datasets (linked to sources)
<a href="https://www.yelp.com/dataset">Visit the Yelp Dataset</a>

### https://www.yelp.com/dataset ###
---
The datasets cover aspects such as business information, customer check-ins, reviews, tips, and user data.
## Project Management

### Overview

| Phase | Details |
| :--- | :--- |
| Data Sourcing | Data was sourced from the [Yelp Dataset](https://www.yelp.com/dataset)   |
| Version Control | Git strategy for collaborative work across team members  |
| EDA | Initial exploration of datasets to understand data structure and identify key variables   
| Merging Datasets | The reviews and business datasets were merged based on the `business_id`. The tips dataset was also considered but was ultimately not merged due to data loss during merging. |
| Preprocessing | Data cleaning, feature selection, and merging of datasets to create a unified analysis dataset  |
| Modeling | Developed and fine-tuned machine learning models, primarily using BERT-based models, to classify and analyze sentiments. |
| Spooder App | Built a web application using Dash to provide an interactive interface for analyzing business reviews. |

### Instillations
Installing necessary libraries 
NOTE: Uncomment any libraries not currently present in your environment for
####  initial execution of this notebook
|  |  |
| :--- | :--- |
 |General utilities| |
|# %pip install pandas --quiet                  |# Data manipulation and analysis|
|# %pip install numpy --quiet                   |# Numerical computations|
|# %pip install scipy --quiet                   |# Scientific computing|
|# %pip install matplotlib --quiet              |# Plotting and visualization|
|# %pip install seaborn --quiet                 |# Statistical data visualization|
|# %pip install tqdm --quiet                    |# Progress bar for loops|
|# %pip install gdown --quiet                   |# Downloading files from Google Drive|
|# %pip install zipfile --quiet                 |# Working with zip files|
|# %pip install json --quiet                    |# JSON handling|

|  |  |
| :--- | :--- |
|Machine Learning & NLP| |
|# %pip install torch --quiet                   |# PyTorch for deep learning|
|# %pip install transformers --quiet            |# HuggingFace Transformers|
|# %pip install datasets --quiet                |# HuggingFace Datasets|
|# %pip install scikit-learn --quiet            |# Machine learning tools|
|# %pip install nltk --quiet                    |# Natural Language Toolkit for text processing|
|# %pip install accelerate --quiet              |# Accelerate training|
|# %pip install evaluate --quiet                |# Metric evaluation|

|  |  |
| :--- | :--- |
|# Web scraping| |
|# %pip install selenium --quiet                |# Browser automation|
|# %pip install webdriver-manager --quiet       |# Manage WebDriver binaries|
|# %pip install beautifulsoup4 --quiet          |# Parsing HTML and XML|

|  |  |
| :--- | :--- |
|# Environment & API| |
|# %pip install python-dotenv --quiet           |# Load environment variables|
|# %pip install langchain --quiet               |# OpenAI LangChain for AI models|

|  |  |
| :--- | :--- |
|# Dash (Web App Framework)| |
|# %pip install dash --quiet                      | # Dash core components|
|# %pip install dash-bootstrap-components --quiet |# Dash Bootstrap components|

|# Plotting & Visualization
|# %pip install plotly --quiet                  |# Interactive graphing library|

|  |  |
| :--- | :--- |
|# Image Handling|
|# %pip install opencv-python-headless --quiet  |# OpenCV for image processing|

### Imports and Dependencies
|  |  |
| :--- | :--- |
|# General Utilities| |
|import pandas as pd               |# Data manipulation and analysis|
|import os                         |# Operating system interfaces|
|import re                         |# Regular expressions|
|import json                       |# JSON handling|
|import time                       |# Time management|
|import zipfile                    |# Working with zip files|
|import unicodedata                |# Unicode character handling|
|import numpy as np                |# Numerical computations|
|import scipy as sp                |# Scientific computing|
|import gdown                      |# Google Drive file download|
|from tqdm import tqdm             |# Progress bar for loops|
|import base64                     |# Encoding and decoding binary data|
|from io import BytesIO            |# Handling binary data in memory|

|  |  |
| :--- | :--- |
|# Image Handling| |
|import cv2                       |# OpenCV for image processing|
|from PIL import Image            | # Image processing via PIL (for handling image conversion)|

|  |  |
| :--- | :--- |
|# Plotting and Visualization| |
|import matplotlib.pyplot as plt   |# Plotting and visualization|
|import matplotlib.ticker as mtick |# Setting ticks to larger numbers|
|import seaborn as sns             |# Statistical data visualization|
|import plotly.express as px       |# Simple interactive plots|
|import plotly.graph_objects as go |# Detailed interactive plots|

|  |  |
| :--- | :--- |
|# Machine Learning & NLP| |
|import torch                                          |# PyTorch for deep learning|
|from sklearn.model_selection import train_test_split  |# Data splitting for training and testing|
|from datasets import load_metric                      |# Compute metrics for NLP models|
|import nltk                                           |# Natural Language Toolkit for text processing|
|from nltk.corpus import stopwords                     |# Stop words for text preprocessing|
|from nltk.tokenize import word_tokenize               |# Tokenization of text|
|import transformers                                   |# HuggingFace Transformers|

|  |  |
| :--- | :--- |
|# Pretrained Model and Tokenization| |
|from transformers import DistilBertForSequenceClassification, DistilBertTokenizer  |# DistilBERT model and tokenizer|
|from transformers import AutoTokenizer, AutoModelForSequenceClassification         |# Auto-tokenizer and model for sequence classification|
|from transformers import DataCollatorWithPadding                                   |# Dynamic padding for batched data|
|from transformers import TrainingArguments, Trainer                                |# Training arguments and trainer|
|from transformers import pipeline                                                  |# Inference pipeline|

|  |  |
| :--- | :--- |
|# Dataset Formatting| |
|import accelerate                           |# Accelerate training|
|from datasets import Dataset                |# Dataset handling|
|from evaluate import load                   |# Metric evaluation|

|  |  |
| :--- | :--- |
|# Web Scraping| |
|from selenium import webdriver                                          |# Browser automation|
|from selenium.webdriver.chrome.service import Service as ChromeService  |# WebDriver service for Chrome|
|from selenium.webdriver.support.ui import WebDriverWait                 |# WebDriver wait|
|from selenium.webdriver.common.by import By                             |# Locating elements by attributes|
|from selenium.webdriver.support import expected_conditions as EC        |# Expected conditions for WebDriver waits|
|from webdriver_manager.chrome import ChromeDriverManager                |# Manage WebDriver binaries|
|from bs4 import BeautifulSoup                                           |# Parsing HTML and XML|

|  |  |
| :--- | :--- |
|# Environment & API| |
|from dotenv import load_dotenv              |# Load environment variables|
|from langchain_openai import ChatOpenAI     |# OpenAI API for LangChain|

|  |  |
| :--- | :--- |
|# Prompt Template and LLM Chain| |
|from langchain import PromptTemplate        |# Prompt template for LangChain|
|from langchain.chains import LLMChain       |# LLM Chain for linking models|

|  |  |
| :--- | :--- |
|# Dash (Web App Framework)| |
|from dash import Dash, dcc, html, callback, callback_context  |# Dash core components and callbacks|
|from dash.dependencies import Input, Output, State            |# Dash dependencies for callbacks|
|from dash.exceptions import PreventUpdate                     |# Prevent updates in callbacks|
|import dash_bootstrap_components as dbc                       |# Dash Bootstrap components|

|  |  |
| :--- | :--- |
|# Other| |
|import math                      |# Mathematical functions|

### Methods
Data Management with gdown
Due to the large size of the training files required for our models, alternative storage solutions to Git were necessary as a result of storage limits. We selected gdown for its integration with Google Drive. 
The notebook json_conversion_for_gdown serves to convert our source files and test functions for retrieval from Google Drive. After successful tests, the relevant code was integrated into our main working notebooks. This notebook is retained in Resources/ for reference.

Original training data sources can be found at https://www.yelp.com/dataset

This project includes multiple components such as data fetching, exploratory data analysis (EDA), data integration, BERT-based Models, web scraping, LangChain, and OpenAI:
- **Data Fetching: Automated scripts to download data from Google Drive.**
- **EDA: Scripts for performing initial exploratory data analysis on the fetched data.**
- **Sentiment Analysis Using BERT-based Models: Use trained model on test dataset to predict sentiments**
- **Universal Business Review Sentiment Analysis**
- **Review Scraping: Scripts to scrape reviews and other details from Google Maps using Selenium and BeautifulSoup.**
- **Apply the Sentiment Analysis Model to the Web Scrapped Data**
- **se Reviews from Selected Business to run ChatGPT Model**
- **Spooder Ap**

---
#### Steps
Detailed Steps
1.	Fetching Data: Use the provided scripts to download datasets from Google Drive. Alternatively, the data can be directly download from https://www.yelp.com/dataset and processed, dependent on the user's computing power and graphic card. Uncomment appropriate code if datasets are download directly. The 5 datasets from Yelp contain Business dataset, Checkin dataset, Reviews dataset, Tips dataset, and User dataset.
   
def fetch_data(set):  # Function to access datasets through `gdown`
________________________________________________________________________________________________________________________________________________________________
2.  Performing EDA: Analyze the data to uncover insights and prepare for further processing.
    We began by loading and examining the following datasets:
- **Business Dataset**: Contains business data including location data, attributes, and categories.
- **Checkin Dataset**: Contains check-in data on businesses (not used in final analysis due to low relevance).
- **Reviews Dataset**: Contains full review text data, along with user and business identifiers.
- **Tips Dataset**: Contains shorter user recommendations on businesses (merged with reviews for analysis).
- **User Dataset**: Contains user data, including friend mappings and metadata (excluded to preserve anonymity).

The data was preprocessed by removing irrelevant columns, renaming variables, and merging relevant datasets for further analysis.  The final merged dataset prepared for machine learning included business id, stars, for rating purposes, review text for NLP tasks, stars average ratings, and other relevant buiness metadata.

### Main Functions:
| Function | Details |
| :--- | :--- |
|fetch_data(set) |Downloads and reads datasets into DataFrames.|
|pd.read_csv() |Reads CSV files into DataFrames.|
|DataFrame.head() |Displays the first few rows of the DataFrame.|
|DataFrame.info() |Provides a summary of the DataFrame, including data types and non-null counts.|
|DataFrame.isna().sum() |Counts missing values in each column.|
|DataFrame.drop() |Drops irrelevant or low-value columns.|
|DataFrame.rename() |Renames columns for standardization.|
|DataFrame.merge() |Merges multiple DataFrames to create a consolidated dataset.|
|seaborn.barplot() |Visualizes missing data percentages.|
|DataFrame.replace() |Combines and reduces the number of categories for target variables.|
_______________________________________________________________________________________________________________________________________________________________
3. Sentiment Analysis Using BERT-based Models
 - **Data Sampling and Preparation**: The dataset was balanced by sampling Yelp reviews based on their star ratings.
 - **Test Preprocessing**: Cleaning and normalizing text data for modal processing.
 - **Model Training**: Pre-trained DistilBERT model for sentiment classification.
 - **Evaluation**: Use metrics accuracy, precision, recall, and F1 score to evaluate model performance.
 - **Model Deployment**: Use trained model on test dataset to predict sentiments.

Star ratings were encoded into 3 categories using label encoding. 
-	0 for negative (1, 2 stars)
-	1 for neutral (3 stars)
-	2 for positive (4, 5 stars)

Sample Evaluation Metrics:
-	Accuracy: 0.8169
-	Precision: 0.8175
-	Recall: 0.8169
-	F1 Score: 0.8171

### Main Functions:
| Function | Details |
| :--- | :--- |
|remove_accented_chars(text)| Removes accented characters from text.|
|clean_text(text)| Cleans text by removing URLs, mentions, hashtags, and extra spaces.|
|pre_process_reviews(reviews) | Preprocesses review text by cleaning and removing stop words.|
|sample_stars(df, val)| Balances dataset by sampling based on star ratings.|
|tokenizer_function(review)| Tokenizes text data for model input.|
|DistilBertForSequenceClassification.from_pretrained()| Loads a pre-trained BERT model for sequence classification.|
|DistilBertTokenizer.from_pretrained()| Loads the tokenizer associated with the BERT model.|
|Dataset.from_dict()| Converts text data into HuggingFace dataset format.|
|TrainingArguments()| Configures training parameters for the model.|
|Trainer()| Initializes the model training setup with data and evaluation metrics.|
|trainer.train()| Fine-tunes the BERT model on the training dataset.|
|compute_metrics(pred)| Computes evaluation metrics (accuracy, precision, recall, F1-score).|
|pipeline('sentiment-analysis')| Creates a sentiment analysis pipeline using the fine-tuned model.|
|apply_roberto(df, review_col)| Applies sentiment analysis to a DataFrame of reviews.|
|model.save_pretrained() and tokenizer.save_pretrained()| Saves the trained model and tokenizer.|
|gdown.download() and zipfile.ZipFile()| Downloads and extracts the pre-trained model from Google Drive.|

____________________________________________________________________________________________________________________________________________________________
4. Universal Business Review Sentiment Analysis
This portion of the application is designed to perform sentiment analysis on business reviews. It classifies reviews into positive, neutral or negative and provides confidence scoress for each classification. While trained on Yelp! data, and developed for Google Reviews, the goal of the application is to be as univerally applicable to business reviews as possible - regardless of the source. 
- **Sentiment Analysis**: Using a pre-trained model nick named "reobert" to analyze sentiment of reviews and assigns a sentiment lable of positive, neutral, or negative.
- **Data Aggregation**: Functions designed to retrieve and process business names and reviews from a dataset.
- **Sentiment Summary**: Classify sentiment for a business based on all availabe reviews.

### Main Functions:
| Function | Details |
| :--- | :--- |
apply_roberto() | Generates sentiment analysis for reviews in a given dataset, and a confidence in that sentiment|
business_names_list() | Generates a list of unique business names from a given dataset|
reviews_list() | Generates a list of all reviews submitted to a business for all its locations|
general_sentiment()	| Classifies the general sentiment for a business' reviews and provides a mean confidence in that sentiment 
Note: To be run after a DataFrame has been passed through apply_roberto()|

Function outlined in more detail below require a DataFrame with the following features:
Feature	Notes
| Column | Details |
| :--- | :--- |
|bus_name_col |A text column with the name of a business|
|bus_add |A text column with the street address of a business' location|
|rev_col |A text column with available reviews|
|sent_lbl|A text column with the generated sentiment classification 
                       Note: Generated through apply_roberto()|
|sent_scr|A text column with the generated sentiment classification
                       Note: Generated through apply_roberto()|
                       
---
5.  Scraping Reviews: Utilize Selenium to scrape reviews from specified Google Maps URLs
ChromeDriver must be installed on your system to run the web scraping for business information, along with pandas, selenium, and beautifulsoup4.

This is a Python-based web scraper designed to extract business details and customer reviews from Google Maps using Selenium and BeautifulSoup. The scraper navigates through multiple Google Maps URLs, collects business information such as names, ratings, addresses, and retrieves customer reviews. The extracted data is then organized into a structured format for further analysis.

Script functions:
- **Setup WebDriver**: Initalizes a Selenium Web to interact with web pages.
- **Navigate and Extract Data**: Accesses specific URLs to extract business details such as business name, average rating, and business address. Captures and parses customer reviews, text and ratings into a DataFrame.
- **Data Handling**: Organizes and stores the extracted data in a structured format using pandas. 
- **Automated Scrolling and Clicks**: Simulates user interction for loading more reviews.

### Main Functions:
| Function | Details |
| :--- | :--- |
|business_Overview(business_name, avg_rating, address1, lat, long, df)  |#Creates a DataFrame with business details such as name, average rating, address, latitude, and longitude.|
|get_review_summary(result_set) |#Extracts review text and ratings from the parsed HTML content and returns a DataFrame with this data.|
|Web Scraping with Selenium  |#Automates the browser to navigate through a list of Google Maps URLs, extract business information, and load reviews.|
|Data Aggregation   |#Combines individual DataFrames from multiple businesses into a single DataFrame (spooder_df) for consolidated data analysis.|

---
6.  Apply the Sentiment Analysis Model to the Web Scrapped Data
The code applies sentiment analysis to Google Reviews data and processes the results to calculate the overall sentiment of a business and compile a list of its reviews. This processed data can then be used for further analysis, reporting, or input into other models, such as generating responses or insights using ChatGPT.

- **Applies Sentiment Analysis**: Uses a sentiment analysis model to analyze the sentiment of each review in the scraped data.
- **Calculates Overall Sentiment**: Aggregates the sentiment analysis results to determine the overall sentiment for specific businesses. This provides a general sentiment score or label that represents the business's customer feedback.
- **Compiles Reviews**: Collects all reviews for a specific business into a list, which can be used for further analysis, reporting, or as input for other models, such as generating responses or insights.

### Main Functions:
| Function | Details |
| :--- | :--- |
|apply_roberto(spooder_df, 'review') |Applies sentiment analysis to reviews and returns a DataFrame with sentiment labels and scores.|
|general_sentiment(roberto_df, 'bus_id', 'Dulce de Leche Bakery', 'sent_label', 'sent_score') |Calculates the overall sentiment for a specific business based on the sentiment labels and scores.|
|reviews_list(roberto_df, 'bus_id', 'Dulce de Leche Bakery', 'bus_add', 'review') |Compiles all reviews for a specific business into a list for further use.|

---
7.  Use Reviews from Selected Business to run ChatGPT Model
This code uses OpenAI's GPT-3.5-turbo model to analyze customer reviews and generate a summary along with actionable recommendations for improving a business.

- **Load Environment and Initialize Model**: Sets up the OpenAI API and initializes the GPT-3.5-turbo model.
- **Generate Prompt**: Creates a prompt template to instruct the model to summarize reviews and suggest improvements.
- **Runs the Language Model Chain**: Uses the LLMChain model to generate a summary and recommendations based on the provided reviews.
- **Extract and Display Results**: Extracts the summary and recommendations from the model's output and prints them to the console.

### Main Functions:
| Function | Details |
| :--- | :--- |
|ChatOpenAI  |Set the model name for our LLMs.|
|PromptTemplate |Construct the prompt template.|
|LLMChain  |Construct a chain using this template.|
|chain.invoke() |Run the chain using the query as input and get the result.|

---
8.  Spooder Ap
    
While trained on Yelp! data, and developed for Google Reviews, the goal of the application is to be as univerally applicable to business reviews as possible - regardless of the source. The following functions were developed with their annotated purposes in mind:

### Main Functions:
| Function | Details |
| :--- | :--- |
|unique_locs_df() |Creates a DataFrame with all unique locations in a given dataset|
|location_details()	|Generates a dictionary with geographic coordinates for all locations of a given business 
Note: To be run on the DataFrame generated by unique_locs_df()|
|build_map() |Constructs a Scattermapbox based on the locations from location_details()|
|apply_davidlingo() |Generates the final summary of a business' reviews, or recommendations for improvement based off the reviews and overall sentiment 
Note: To be used with the ouputs of reviews_list() and general_sentiment()|

Each function outlined in more detail below requires a DataFrame with the following features:
| Column | Details |
| :--- | :--- |
|bus_name_col |A text column with the name of a business|
|bus_add |A text column with the street address of a business' location|
|bus_lat |A float column with the latitude coordinate for a business' location|
|bus_lon |A float column with the longitude coordinate for a business' location|
|rev_col |A text column with available reviews|
|sent_lbl |A text column with the generated sentiment classification 
Note: Generated through apply_roberto()|
|sent_scr |A text column with the generated sentiment classification 
Note: Generated through apply_roberto()|

---
Instructions for Running the Code:
1.  Clone repository to a local directory.
2.  Ensure necessary API Keys are present in your local `.env`.
3.  Uncomment neccesary `pip install`s as needed from `[notebook]`.
4.  If executing for the first time, uncomment any `gdown` fetch cells and comment out their corresponding `.csv` read ins.
If, however, you are executing the notebook any subsequent time, comment out all instances of `gdown` fetch requests, and instead utilize their corresponding `.csv` read ins.
5.  `Run All Cells` to execute the code in every cell of the notebook in sequence. Alternatively, each cell may be `Run` on its own, though it is still recommended to run them in order.  *Note: This will launch the application* `SpooderApp™` *in your default browser.*
6.  Consult onscreen guide in `SpooderApp™` interface for use.

## Findings


![Distribution Image](https://github.com/MxOdele/AI_Project_3_Group_01/tree/main/Images/Disribution_of_Star_Ratings.png "Distribution of Star Ratings")

![Distribution Top 10 Image](https://github.com/MxOdele/AI_Project_3_Group_01/tree/main/Images/Disribution_of_Star_Ratings_Top_10_Business.png "Distribution of Star Ratings Top 10")

![Distribution Review Counts Image](https://github.com/MxOdele/AI_Project_3_Group_01/tree/main/Images/Distribution_of_Review_Counts.png "Distribution of Review Counts")


![Distribution Sentiment Labels Image](https://github.com/MxOdele/AI_Project_3_Group_01/tree/main/Images/Distribution_of_Sentiment_Labels.png "Distribution of Sentiment Labels")

![Distribution Sentiment Scores Image](https://github.com/MxOdele/AI_Project_3_Group_01/tree/main/Images/Distribution_of_Sentiment_Scores.png "Distribution of Sentiment Scores")

![Distribution Top 20 Image](https://github.com/MxOdele/AI_Project_3_Group_01/tree/main/Images/Distribution_of_Top_20_Business_Categories.png "Distribution of Top 20 Business Categories")

![Distribution NA Percentage Plot Image](https://github.com/MxOdele/AI_Project_3_Group_01/tree/main/Images/NA_Percentage_Plot.png "NA Percentage Plot")

![Distribution Sentiment Analysis Image](https://github.com/MxOdele/AI_Project_3_Group_01/tree/main/Images/Sentiment_Analysis_img.png "Sentiment Analysis")

![is_open Image](https://github.com/MxOdele/AI_Project_3_Group_01/tree/main/Images/is_open_Feature_Count.png "is open")

![SpooderApp Image](https://github.com/MxOdele/AI_Project_3_Group_01/tree/main/Images/SpooderApp_Logo_Inverted_Color.png "SpooderApp")

![Yelp&Google Image](https://github.com/MxOdele/AI_Project_3_Group_01/tree/main/Images/yelp&google_AI_generated_img.png "Yelp&Google")


### Results
- **Sentiment Classification Accuracy**: The final model, roberto achieved an accuracy  over 82% in classifying sentiments as positive, negative, or neutral
- **Actionable Feedback**: The OpenAI LangChain model, davidlingo, effectively summarizes available reviews and provides consumer-driven recommendations for improvements to operations, regardless of business scale.
- **User Engagement and interaction**: The deployment of the sentiment analysis model into a Dash web application allowed users to interactively explore data. Feedback from user sessions can highlight the application's utility in providing immediate sentiment insights, which can be particualry useful for business owners and manager.

### Conclusion
-**Practical Benefits for Businesses**: The sentiment analysis tool provides businesses with a practical way to track and improve customer service by focusing on key factors that affect customer satisfaction. By incorporating sentiment analysis into their customer relationship management (CRM) systems, businesses can actively manage their reputations and strengthen customer loyalty.

-**Improving Customer Experience**: The analysis delivers valuable insights that enable businesses to better customize their services or products to align with customer expectations.
Placeholder

### Future Considerations
If given more time to develop its features further, **SpooderApp™** was envisioned to meet the following milestones;
- Allow for ad hoc searches of businesses and real-time web scraping
- Retrieve a complete list of available reviews in lieu of the sampled up-to-fifty (=< 50)
- Train `roberto` on a larger dataset than the six thousand (6,000) reviews provided
- Brand and logo development (because a cute little spooder logo would be too adorable for words)

---

## Citations and Licenses

Sources
•	The data and analysis code in this project are licensed under the MIT License.

•   Yelp Dataset [Yelp Open Dataset](https://www.yelp.com/dataset)
[^1]: Image source