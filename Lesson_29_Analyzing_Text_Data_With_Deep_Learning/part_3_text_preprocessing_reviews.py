import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv('reviews.csv')

# Initialize the stemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Removing stop words and stemming
    processed_tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(processed_tokens)

data['cleaned_reviews'] = data['reviews'].apply(preprocess_text)
