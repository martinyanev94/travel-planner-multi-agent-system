from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# nltk.download('punkt') # Uncomment if necessary for tokenization

# Sample text corpus
corpus = [
    "Cats are great pets.",
    "Dogs are also great companions.",
    "I love my cat.",
    "My dog loves to play outside.",
    "Pets can be incredibly loving."
]

# Tokenizing the sentences
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Training the Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=10, window=2, min_count=1, workers=4)

# Accessing the vector for a word
cat_vector = model.wv['cat']
print(f"Vector representation for 'cat': {cat_vector}")
