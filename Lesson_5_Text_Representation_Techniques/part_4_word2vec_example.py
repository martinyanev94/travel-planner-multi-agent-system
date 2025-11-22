from gensim.models import Word2Vec
import numpy as np

# Sample text data
sentences = [
    ["I", "love", "programming"],
    ["Python", "is", "great"],
    ["I", "enjoy", "learning"],
]

# Create Word2Vec model
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=4)

# Accessing the vector for the word "Python"
vector = model.wv["Python"]
print("Word Vector for 'Python':", vector)
