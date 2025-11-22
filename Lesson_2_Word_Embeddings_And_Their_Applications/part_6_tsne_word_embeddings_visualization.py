from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Preparing the embeddings and labels
words = list(glove_model.keys())
embeddings = np.array([glove_model[word] for word in words])

# Fit and transform with t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
reduced_embeddings = tsne.fit_transform(embeddings)

# Create scatter plot
plt.figure(figsize=(12, 12))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)

for i, word in enumerate(words):
    plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

plt.title("t-SNE visualization of word embeddings")
plt.show()
