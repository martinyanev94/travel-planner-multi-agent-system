from gensim.models import Word2Vec

# Sample sentence data
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["dogs", "are", "great", "pets"],
    ["the", "cat", "and", "the", "dog", "are", "friends"]
]

# Create and train the Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Getting the vector for a specific word
cat_vector = model.wv["cat"]
print("Vector representation of 'cat':")
print(cat_vector)
