similar_words = model.wv.most_similar('cat', topn=3)
print("Words most similar to 'cat':", similar_words)
