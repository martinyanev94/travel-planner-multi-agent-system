from sklearn.metrics.pairwise import cosine_similarity

def get_embedding_similarity(word1, word2, model, tokenizer):
    word1_index = tokenizer.word_index[word1]
    word2_index = tokenizer.word_index[word2]
    
    word1_embedding = model.layers[0].get_weights()[0][word1_index]
    word2_embedding = model.layers[0].get_weights()[0][word2_index]
    
    similarity = cosine_similarity([word1_embedding], [word2_embedding])
    return similarity[0][0]

# Compare similarities
word_a = "cat"
word_b = predicted_context
similarity_score = get_embedding_similarity(word_a, word_b, model, tokenizer)

print(f"The cosine similarity between '{word_a}' and '{word_b}' is {similarity_score:.4f}.")
