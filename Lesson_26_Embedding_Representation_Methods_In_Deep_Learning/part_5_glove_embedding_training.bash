pip install glove-python-binary
from glove import Corpus, Glove

# Assume we already have a list of sentences for training
# Using a smaller sample for explanation. A real-world scenario would use a larger set.

sentences = [
    ['i', 'love', 'machine', 'learning'],
    ['deep', 'learning', 'is', 'a', 'part', 'of', 'machine', 'learning'],
    ['word', 'embeddings', 'are', 'useful', 'in', 'nlp'],
    ['i', 'also', 'enjoy', 'artificial', 'intelligence']
]

corpus = Corpus()
corpus.fit(sentences, window=10)

glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.word_counts, epochs=30, no_threads=4, verbose=True)
glove.add_display_words(['learning', 'nlp', 'artificial', 'intelligence'])

# Accessing the vector for the word 'learning':
print(glove.word_vectors[glove.dictionary['learning']])
