pip install gensim
from gensim.models import Word2Vec

# Sample sentences for training the Word2Vec model
sentences = [
    ['the', 'cat', 'sits', 'on', 'the', 'mat'],
    ['the', 'dog', 'plays', 'with', 'the', 'ball'],
    ['the', 'cat', 'and', 'dog', 'are', 'friends']
]

# Create and train the Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

# Get the vector for a specific word
cat_vector = model.wv['cat']
print(cat_vector)
