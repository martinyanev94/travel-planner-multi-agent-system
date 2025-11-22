def get_embedding_matrix(tokens, embeddings_index, embedding_dim=100):
    embedding_matrix = np.zeros((len(tokens), embedding_dim))
    for i, word in enumerate(tokens):
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
    return embedding_matrix

# Example usage
embedding_matrix = get_embedding_matrix(tokens, embeddings_index)
print(embedding_matrix)
