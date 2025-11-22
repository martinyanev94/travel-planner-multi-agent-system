pip install gensim
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')

# Sample corpus
corpus = [
    "Natural language processing is a fascinating field.",
    "Word embeddings allow words to be represented as vectors.",
    "This enables better understanding of textual data."
]

# Tokenizing the sentences
sentences = [nltk.word_tokenize(sentence.lower()) for sentence in corpus]

# Creating the Word2Vec model
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=4)

# Getting vector for a word
word_vector = model.wv['language']
print(word_vector)
