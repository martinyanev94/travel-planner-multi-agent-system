from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Get feature names and resultant TF-IDF matrix
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_array = tfidf_matrix.toarray()

print("TF-IDF Feature Names:", tfidf_feature_names)
print("TF-IDF Matrix:")
print(tfidf_array)
