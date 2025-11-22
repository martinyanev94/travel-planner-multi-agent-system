from numpy import dot
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Calculate similarity between 'cat' and 'dog'
similarity_score = cosine_similarity(model.wv['cat'], model.wv['dog'])
print(f"Cosine similarity between 'cat' and 'dog': {similarity_score}")
