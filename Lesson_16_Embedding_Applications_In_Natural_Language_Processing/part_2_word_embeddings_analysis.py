# Loading the model
model = Word2Vec.load("word2vec.model")

# Finding similar words
similar_words = model.wv.most_similar("natural", topn=5)
for word, similarity in similar_words:
    print(f"Word: {word}, Similarity: {similarity:.4f}")
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "Natural language processing enables machines to understand human language.",
    "AI is a field of study that focuses on creating intelligent agents.",
    "Deep learning is a subset of machine learning.",
    "Language models based on deep learning have become quite popular."
]

# Applying TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Displaying the TF-IDF matrix
import pandas as pd

df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
print(df_tfidf)
