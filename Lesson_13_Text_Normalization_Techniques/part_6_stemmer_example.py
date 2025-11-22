from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in cleaned_text.split()]
print(stemmed_tokens)
