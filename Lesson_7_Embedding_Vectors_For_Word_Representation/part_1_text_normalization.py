import re
import nltk
from nltk.corpus import stopwords

# Ensure you have the NLTK stop words
nltk.download('stopwords')

def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text
    tokens = text.split()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

sample_text = "The sun shines brightly in the sky"
normalized_text = normalize_text(sample_text)
print(normalized_text)  # Output: sun shines brightly sky
