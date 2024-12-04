# Spam-mail-classification

This project implements a Spam Email Classification system using the Naive Bayes algorithm. The system classifies emails as either spam or ham (non-spam) based on the content of the emails. The project uses TF-IDF vectorizer for feature extraction, and the trained model is saved using Pickle for future use. A Streamlit app is included for easy interaction with the model.

## Features

- **Spam Detection:** Classifies emails as spam or ham using a Naive Bayes classifier.
- **TF-IDF Vectorization:** Converts text data into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF) method.
- **Model Persistence:** The trained model is saved using Pickle, so it doesn't need to be retrained each time.
- **Streamlit App:** A user-friendly web interface to test the classifier in real-time.

## Requirements

To run this project, you will need the following Python libraries:

- pandas
- numpy
- scikit-learn
- nltk
- pickle
- streamlit

You can install the required libraries using pip:

```bash
pip install -r requirements.txt
