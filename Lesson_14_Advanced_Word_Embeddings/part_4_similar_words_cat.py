similar_words = model.wv.most_similar('cat', topn=3)
print(f"Words similar to 'cat': {similar_words}")
