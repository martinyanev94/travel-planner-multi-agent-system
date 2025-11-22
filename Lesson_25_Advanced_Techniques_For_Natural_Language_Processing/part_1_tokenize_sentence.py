import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')  # Download the Punkt tokenizer models

sentence = "Natural Language Processing enables computers to understand human language."
tokens = word_tokenize(sentence)
print(tokens)
