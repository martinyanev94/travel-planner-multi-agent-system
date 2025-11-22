pip install gensim
from gensim.models import Word2Vec

# Sample corpus
sentences = [
    ["I", "love", "natural", "language", "processing"],
    ["Natural", "language", "processing", "is", "fun"],
    ["I", "enjoy", "learning", "about", "AI"],
    ["Machine", "learning", "is", "a", "subset", "of", "AI"]
]

# Training the Word2Vec model
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, workers=4)

# Save the model to disk
model.save("word2vec.model")
