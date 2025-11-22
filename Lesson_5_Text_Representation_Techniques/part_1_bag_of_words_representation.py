from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love programming in Python",
    "Python programming is fun",
    "I enjoy learning new programming languages",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Bag of Words Representation:\n", X.toarray())
