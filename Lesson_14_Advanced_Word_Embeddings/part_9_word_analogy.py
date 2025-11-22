analogy = model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=3)
print(f"Word analogy result: {analogy}")
