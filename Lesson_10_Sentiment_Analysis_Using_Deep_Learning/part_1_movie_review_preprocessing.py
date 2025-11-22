import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
url = "https://raw.githubusercontent.com/laxmimerit/movie_review_dataset/main/movie.csv"
data = pd.read_csv(url)

# Display the first few rows
print(data.head())
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure you have the nltk stopwords downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

data['cleaned_reviews'] = data['review'].apply(preprocess_text)

# Display the cleaned reviews
print(data[['review', 'cleaned_reviews']].head())
