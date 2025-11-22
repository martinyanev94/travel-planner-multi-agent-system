!pip install gensim
from gensim.models import Word2Vec
from nltk.corpus import brown
import nltk

nltk.download('brown')

# Load the Brown Corpus
sentences = brown.sents()

# Train the Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# Save the model
model.save("brown_word2vec.model")
