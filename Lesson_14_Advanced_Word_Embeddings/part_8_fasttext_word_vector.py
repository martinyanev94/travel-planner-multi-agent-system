from gensim.models import FastText

# Training FastText model
fasttext_model = FastText(sentences=tokenized_corpus, vector_size=10, window=3, min_count=1, workers=4)

# Accessing the vector for a word
fish_vector = fasttext_model.wv['fish']
print(f"Vector representation for 'fish': {fish_vector}")
