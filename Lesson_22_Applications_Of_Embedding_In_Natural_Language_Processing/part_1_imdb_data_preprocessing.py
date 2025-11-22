from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

# Set parameters
max_words = 10000  # Considering only the top 10,000 most common words
max_len = 100  # We will pad the sequences to a maximum length of 100

# Load and split the dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# Pad sequences to ensure they are the same length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
