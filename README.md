# AI_Project_3_Group_01
A data analysis class project 3 for Columbia University's AI bootcamp.

## Title Placeholder

<div align='center'>
    <img src='' height='300' title='Placeholder' alt='Placeholder' />
## Yelp Business and Review Analysis ##
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
| App | Built a web application using Dash to provide an interactive interface for analyzing business reviews. |

### Dependencies
List of Python packages required:
- **python 3.x
- **pandas
- **numpy
- **scipy
- **nltk
- **scikit-learn
- **transformers
- **evaluate
- **torch
- **tqdm
- **accelerate
- **huggingface_hub
- **datasets
- **matplotlib
- **seaborn
- **selenium
- **beautifulsoup4
- **gdown
- **webdriver_manager
- **dash
- **json
Other items:
- **Google Chrome
- **.env file with appropriate API Keys for ChapGPT
- **Prepared csv file with URLs of businessnes, latitude and longitude.
- **Zipfile

### Methods
Data Management with gdown
Due to the large size of the training files required for our models, alternative storage solutions to Git were necessary as a result of storage limits. We selected gdown for its integration with Google Drive. 
The notebook json_conversion_for_gdown serves to convert our source files and test functions for retrieval from Google Drive. After successful tests, the relevant code was integrated into our main working notebooks. This notebook is retained in Resources/ for reference.

Original training data sources can be found at https://www.yelp.com/dataset

This project includes multiple components such as data fetching, exploratory data analysis (EDA), and data integration:
•	Data Fetching: Automated scripts to download data from Google Drive.
•	EDA: Scripts for performing initial exploratory data analysis on the fetched data.
•	Review Scraping: Scripts to scrape reviews and other details from Google Maps using Selenium and BeautifulSoup.

#### Placeholder for a step
Detailed Steps
1.	Fetching Data: Use the provided scripts to download datasets from Google Drive. Alternatively, the the data can be directly download from https://www.yelp.com/dataset and processed, dependent on the user's computing power and graphic card. Uncomment appropriate code if datasets are download directly. The 5 datasets from Yelp contain Business dataset, Checkin dataset, Reviews dataset, Tips dataset, and User dataset.
   
def fetch_data(set):  # Function to access datasets through `gdown`

2.  Performing EDA: Analyze the data to uncover insights and prepare for further processing.
    We began by loading and examining the following datasets:
- **Business Dataset**: Contains business data including location data, attributes, and categories.
- **Checkin Dataset**: Contains check-in data on businesses (not used in final analysis due to low relevance).
- **Reviews Dataset**: Contains full review text data, along with user and business identifiers.
- **Tips Dataset**: Contains shorter user recommendations on businesses (merged with reviews for analysis).
- **User Dataset**: Contains user data, including friend mappings and metadata (excluded to preserve anonymity).

The data was preprocessed by removing irrelevant columns, renaming variables, and merging relevant datasets for further analysis.  The final merged dataset prepared for machine learning included business id, stars, for rating purposes, review text for NLP tasks, stars average ratings, and other relevant buiness metadata.


3. Sentiment Analysis Using BERT-based Models
 - **Data Sampling and Preparation**: The dataset was balanced by sampling Yelp reviews based on their star ratings.
 - **Test Preprocessing**: Cleaning and normalizing text data for modal processing.
 - **Model Training**: Pre-trained DistilBERT model for sentiment classification.
 - **Evaluation**: Use metrics accuracy, precision, recall, and F1 score to evaluate model performance.
 - **Model Deployment**: Use trained model on test dataset to predict sentiments.

Star ratings were encoded into 3 categories using label encoding. 
•	0 for negative (1, 2 stars)
•	1 for neutral (3 stars)
•	2 for positive (4, 5 stars)

Sample Evaluation Metrics:
•	Accuracy: 0.8169
•	Precision: 0.8175
•	Recall: 0.8169
•	F1 Score: 0.8171

4. Universal Business Review Sentiment Analysis

This portion of the application is designed to perform sentiment analysis on business reviews. It classifies reviews into positive, neutral or negative and provides confidence scoress for each classification.
- **Sentiment Analysis**: Using a pre-trained model nick named "reobert" to analyze sentiment of reviews and assigns a sentiment lable of positive, neutral, or negative.
- **Data Aggregation**: Functions designed to retrieve and process business names and reviews from a dataset.
- **Sentiment Summary**: Classify sentiment for a business based on all availabe reviews.


5.  Scraping Reviews: Utilize Selenium to scrape reviews from specified Google Maps URLs
ChromeDriver must be installed on your system to run the web scraping for business information, along with pandas, selenium, and beautifulsoup4.

This is a Python-based web scraper designed to extract business details and customer reviews from Google Maps using Selenium and BeautifulSoup. The scraper navigates through multiple Google Maps URLs, collects business information such as names, ratings, addresses, and retrieves customer reviews. The extracted data is then organized into a structured format for further analysis.

Script functions:
- **Setup WebDriver**: Initalizes a Selenium Web to interact with web pages.
- **Navigate and Extract Data**: Accesses specific URLs to extract business details such as business name, average rating, and business address. Captures and parses customer reviews, text and ratings into a DataFrame.
- **Data Handling**: Organizes and stores the extracted data in a structured format using pandas. 
- **Automated Scrolling and Clicks**: Simulates user interction for loading more reviews.
    
---

## Findings

### Results

Placeholder

### Conclusion

Placeholder

### Future Considerations

Placeholder

---

## Citations and Licenses

Sources
•	The data and analysis code in this project are licensed under the MIT License.

•   Yelp Dataset [Yelp Open Dataset](https://www.yelp.com/dataset)
[^1]: Image source