# Load the model
model = Word2Vec.load("brown_word2vec.model")

# Find similar words for 'king'
similar_words = model.wv.most_similar('king', topn=5)
print("Words similar to 'king':", similar_words)
