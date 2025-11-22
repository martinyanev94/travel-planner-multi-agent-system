pip install gensim
from gensim.models import Word2Vec

sentences = [
    ['i', 'love', 'machine', 'learning'],
    ['deep', 'learning', 'is', 'a', 'part', 'of', 'machine', 'learning'],
    ['word', 'embeddings', 'are', 'useful', 'in', 'nlp'],
    ['i', 'also', 'enjoy', 'artificial', 'intelligence']
]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)
word_vector = model.wv['learning']
print(word_vector)
