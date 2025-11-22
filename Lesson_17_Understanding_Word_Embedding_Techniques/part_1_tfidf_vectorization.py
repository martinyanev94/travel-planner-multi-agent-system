from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "The cat sat on the mat.",
    "Dogs are great pets.",
    "The cat and the dog are friends."
]

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert the result to a dense format and display it
tfidf_dense = tfidf_matrix.todense()
print(tfidf_dense)
print(vectorizer.get_feature_names_out())
