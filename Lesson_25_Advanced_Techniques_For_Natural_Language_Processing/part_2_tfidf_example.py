from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "I love programming.",
    "Programming in Python is great.",
    "I love Python."
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
print(tfidf_matrix.toarray())
print(vectorizer.get_feature_names_out())
