import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

raw_text = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(raw_text.lower())  # Convert to lowercase and tokenize
print("Tokenized words:", tokens)
