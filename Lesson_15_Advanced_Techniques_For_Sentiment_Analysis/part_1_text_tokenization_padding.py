import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Sample data
texts = ['I love this movie', 'This film is terrible', 'What a fantastic performance', 'I did not enjoy it at all']
labels = [1, 0, 1, 0]  # 1 for positive sentiment, 0 for negative sentiment

# Tokenizing the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding sequences
max_length = max(len(x) for x in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

print("Padded Sequences:")
print(padded_sequences)
