import numpy as np

# Sample word embeddings
word_embeddings = {
    'king': np.array([0.5, 0.6]),
    'queen': np.array([0.4, 0.8]),
    'man': np.array([0.2, 0.3]),
    'woman': np.array([0.1, 0.5]),
}

# Function to find the relationship between words
def get_relationship(word1, word2, word3):
    vector1 = word_embeddings[word1]
    vector2 = word_embeddings[word2]
    vector3 = word_embeddings[word3]
    return vector2 - vector1 + vector3

# Finding the relationship
result_vector = get_relationship('king', 'queen', 'man')
print(f'The result vector represents: {result_vector}')
