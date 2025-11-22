from gensim.models import Word2Vec

# Sample text corpus
sentences = [
    ['the', 'cat', 'sat', 'on', 'the', 'mat'],
    ['dogs', 'are', 'better', 'than', 'cats'],
    ['cats', 'are', 'great', 'pets'],
]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=4)

# Retrieve the embedding for the word 'cat'
cat_vector = model.wv['cat']
print(f'Embedding for "cat": {cat_vector}')
