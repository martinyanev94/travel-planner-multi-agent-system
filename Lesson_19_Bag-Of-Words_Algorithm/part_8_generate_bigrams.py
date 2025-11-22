# Initialize CountVectorizer with bigrams
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))

# Fit and transform the documents
bigram_matrix = bigram_vectorizer.fit_transform(documents)

# Get feature names and resultant bigram matrix
bigram_feature_names = bigram_vectorizer.get_feature_names_out()
bigram_array = bigram_matrix.toarray()

print("Bigram Feature Names:", bigram_feature_names)
print("Bigram Document-Term Matrix:")
print(bigram_array)
