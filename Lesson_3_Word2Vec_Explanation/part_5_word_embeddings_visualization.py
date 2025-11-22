!pip install matplotlib sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_words(model):
    labels = []
    embeddings = []

    for word in model.wv.key_to_index:  # Iterate over words in the vocabulary
        labels.append(word)
        embeddings.append(model.wv[word])

    # Reduce dimensionality to 2D using t-SNE
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

    for i, label in enumerate(labels):
        plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    plt.title("Word2Vec Word Embeddings Visualization")
    plt.show()
