import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Prepare the word vectors
words = list(model.wv.index_to_key)
word_vectors = np.array([model.wv[word] for word in words])

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
result = tsne.fit_transform(word_vectors)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(result[:, 0], result[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.title("t-SNE Visualization of Word2Vec Embeddings")
plt.show()
