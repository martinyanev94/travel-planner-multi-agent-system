# Example of using financial text for training
financial_sentences = [["investment", "bank", "stock", "market"], 
                       ["capital", "asset", "debt", "equity"], 
                       ["financial", "analysis", "report"]]

financial_model = Word2Vec(financial_sentences, vector_size=100, window=2, min_count=1, sg=1)

similar_financial_terms = financial_model.wv.most_similar('bank', topn=5)
print("Financial context similar words to 'bank':", similar_financial_terms)
