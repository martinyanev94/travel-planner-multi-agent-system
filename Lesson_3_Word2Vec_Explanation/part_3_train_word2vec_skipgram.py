# Train Word2Vec model using Skip-gram
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, sg=1, epochs=100)
word_vector = model.wv['cat']
print(f"Vector representation of 'cat': {word_vector}")
