from nltk.corpus import stopwords

# Load stop words
stop_words = set(stopwords.words('english'))

# Filtering stop words
filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words]
print(filtered_tokens)
