import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Sample data: sentences and their corresponding labels
sentences = [
    "I love this product", 
    "This is the worst experience", 
    "I am very happy with my purchase",
    "I will never buy this again"
]
labels = [1, 0, 1, 0]  # 1: Positive, 0: Negative

# Tokenize and pad sequences
tokenized_sentences = [sentence.split() for sentence in sentences]
dictionary = {}
for sentence in tokenized_sentences:
    for word in sentence:
        if word not in dictionary:
            dictionary[word] = len(dictionary) + 1

encoded_sentences = [[dictionary[word] for word in sentence] for sentence in tokenized_sentences]
padded_sentences = pad_sequences(encoded_sentences, padding='post')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(padded_sentences, labels, test_size=0.2)

# Build the RNN model
model = Sequential()
model.add(Embedding(input_dim=len(dictionary) + 1, output_dim=8, input_length=padded_sentences.shape[1]))
model.add(SimpleRNN(32)) 
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")
