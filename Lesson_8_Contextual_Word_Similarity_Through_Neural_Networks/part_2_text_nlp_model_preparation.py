import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Sample sentences
sentences = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the mouse",
    "the dog chased the cat",
    "fish swim in the water"
]

# Prepare the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

# Create input-output pairs for the neural network
input_words = []
output_words = []

for sentence in sentences:
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_list) - 1):
        n_gram_sequence = token_list[i-1:i+2]
        input_words.append(n_gram_sequence[:-1])
        output_words.append(n_gram_sequence[-1])

input_words = np.array(input_words)
output_words = np.array(output_words)

# Prepare the data for training
X_train, X_test, y_train, y_test = train_test_split(input_words, output_words, test_size=0.2)
