def evaluate_bias(word1, word2):
    similarity_score = model.wv.similarity(word1, word2)
    print(f"Similarity between {word1} and {word2}: {similarity_score}")

evaluate_bias("nurse", "woman")
evaluate_bias("doctor", "man")
