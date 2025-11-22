from numpy.linalg import norm

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))

# Example vectors
vector_a = model.wv['king']
vector_b = model.wv['queen']

similarity = cosine_similarity(vector_a, vector_b)
print(f'Cosine similarity between "king" and "queen": {similarity}')
