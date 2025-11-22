from gensim.models import Word2Vec

sentences = [
    ['I', 'love', 'to', 'play', 'football'],
    ['I', 'play', 'tennis', 'and', 'football'],
    ['I', 'love', 'coding', 'in', 'Python']
]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
word_vector = model.wv['football']
print(word_vector)
