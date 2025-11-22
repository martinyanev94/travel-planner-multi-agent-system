from gensim.models import Word2Vec
import nltk
nltk.download('punkt')

# Sample sentences
sentences = [
    "Artificial intelligence is reshaping industries.",
    "Natural language processing bridges human language and technology.",
    "Word embeddings play a significant role in AI applications."
]

# Tokenizing the sentences
tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]

# Training a Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=50, window=2, min_count=1, workers=4)

# Exploring word embeddings
print("Embedding for 'artificial':", model.wv['artificial'])
print("Embedding for 'language':", model.wv['language'])
