from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Simulate some word embeddings for visualization
labels = list(word_embeddings.keys())
embeds = np.array(list(word_embeddings.values()))

# Fit PCA model
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeds)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1])

for i, label in enumerate(labels):
    plt.annotate(label, (pca_result[i, 0], pca_result[i, 1]))

plt.title('PCA of Word Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid()
plt.show()
