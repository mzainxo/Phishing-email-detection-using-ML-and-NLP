# Phishing Email Detection System

This project uses Natural Language Processing (NLP) and Machine Learning techniques to detect phishing emails. By analyzing the contents of emails, the system classifies them as either "Safe" or "Phishing". The model uses **Random Forest** for classification and utilizes **TF-IDF** for text feature extraction.

## Table of Contents

1. [Project Description](#project-description)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Structure](#structure)
6. [License](#license)

## Project Description

Phishing attacks are one of the most common cyber threats that target individuals and organizations. This project aims to develop a reliable system to detect phishing emails using various techniques in Natural Language Processing (NLP) and Machine Learning (ML). The model is trained using a labeled dataset of emails and uses a Random Forest classifier to make predictions.

The core of the system involves:
- **TF-IDF Vectorization**: Transforms the email body text into numerical features that can be used by the machine learning model.
- **Random Forest Classifier**: A robust algorithm for classification based on multiple decision trees.
- **Sentiment Analysis**: Analyzes the sentiment of email content to assist in classification.

The system can be deployed as a web application using **Streamlit**, where users can input email details to get real-time classification results.

## Technologies Used

- **Python**: The main programming language used for the project.
- **Scikit-Learn**: For machine learning models, including Random Forest classifier.
- **NLTK (Natural Language Toolkit)**: For text preprocessing and sentiment analysis.
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: A feature extraction technique used for text classification.
- **Streamlit**: A framework to create the web interface for interacting with the phishing detection system.
- **BeautifulSoup**: For cleaning and extracting text from HTML email bodies.
- **IMAP**: For fetching emails from Gmail for real-time classification.

