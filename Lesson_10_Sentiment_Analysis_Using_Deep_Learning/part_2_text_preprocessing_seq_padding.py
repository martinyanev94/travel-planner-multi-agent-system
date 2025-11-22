from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Tokenize the text
tokenizer = Tokenizer(num_words=5000) # limit to 5000 words for simplicity
tokenizer.fit_on_texts(data['cleaned_reviews'])
sequences = tokenizer.texts_to_sequences(data['cleaned_reviews'])
word_index = tokenizer.word_index

# Pad sequences to ensure uniform input size
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Prepare labels
labels = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
