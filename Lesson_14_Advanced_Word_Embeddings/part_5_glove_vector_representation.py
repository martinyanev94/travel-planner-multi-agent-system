from glove import Corpus, Glove

# Preparing the corpus for GloVe
corpus_model = Corpus()
corpus_model.fit(tokenized_corpus, window=10)

# Creating the GloVe model
glove = Glove(no_components=10, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=100, no_threads=4, verbose=True)

# Accessing the vector for a word
dog_vector = glove.word_vectors[glove.dictionary['dog']]
print(f"Vector representation for 'dog': {dog_vector}")
