from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Example documents
documents = [
    "Natural language processing is fascinating.",
    "Machine learning provides powerful tools.",
    "Deep learning is a subset of machine learning."
]

# User query
query = "What is natural language processing?"

# Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents + [query])

# Compute similarity scores
query_vector = tfidf_matrix[-1, :]  # Last row is the query
similarity_scores = np.dot(tfidf_matrix[:-1], query_vector.T).toarray()
sorted_indices = np.argsort(-similarity_scores.flatten())

# Display sorted documents based on similarity
for index in sorted_indices:
    print(f"Document: {documents[index]}, Similarity Score: {similarity_scores[index][0]}")
