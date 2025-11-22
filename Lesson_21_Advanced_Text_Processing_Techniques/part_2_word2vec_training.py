from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Sample sentences for training the Word2Vec model
sentences = [
    "Natural language processing enables computers to understand human language.",
    "Word embeddings are powerful tools in machine learning.",
    "Understanding context is key to deciphering meaning."
]

# Tokenizing the sentences into words
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Training the Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Getting vector representation for the word 'language'
vector = model.wv['language']
print(vector)
