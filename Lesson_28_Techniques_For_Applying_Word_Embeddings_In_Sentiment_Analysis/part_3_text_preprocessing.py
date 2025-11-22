import re
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize the text into words
    tokens = word_tokenize(text)
    return tokens

# Example usage
sample_text = "Absolutely loved the movie! It was fantastic."
tokens = preprocess_text(sample_text)
print(tokens)
