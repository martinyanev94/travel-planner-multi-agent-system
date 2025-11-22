# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the documents
X = vectorizer.fit_transform(documents)

# Get feature names and the resultant matrix
feature_names = vectorizer.get_feature_names_out()
matrix = X.toarray()

print("Feature Names:", feature_names)
print("Document-Term Matrix:")
print(matrix)
Document-Term Matrix:
[[0 0 1 0 0 0 0 1 1 1]
 [0 0 0 0 1 0 1 0 1 1]
 [1 1 0 1 0 1 0 0 0 0]]
