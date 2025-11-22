pip install glove-python-binary
from glove import Corpus, Glove

# Preparing the data
corpus = Corpus()
corpus.fit(sentences, window=5)

# Creating and training the GloVe model
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

# Get the vector for 'cat'
cat_glove_vector = glove.word_vectors[glove.dictionary['cat']]
print(cat_glove_vector)
