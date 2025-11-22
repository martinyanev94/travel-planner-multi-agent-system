from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Get vector for 'king'
king_vector = glove_model['king']

# Find and calculate cosine similarity for other words
similar_words = {}
for word, vector in glove_model.items():
    similarity = cosine_similarity(king_vector, vector)
    if word != 'king':  # prevent comparing the word to itself
        similar_words[word] = similarity

# Sort by similarity
sorted_similar_words = sorted(similar_words.items(), key=lambda item: item[1], reverse=True)
top_similar = sorted_similar_words[:5]
print("Top similar words to 'king':", top_similar)
