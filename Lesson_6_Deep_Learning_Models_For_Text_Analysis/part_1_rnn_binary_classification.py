from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.preprocessing.sequence import pad_sequences

# Sample data: sequences of integers representing words
X = [[1, 2, 3], [4, 5, 6, 1], [7, 8, 2, 3, 1]]
y = [0, 1, 0]  # Binary classification labels

# Pad sequences to ensure uniform input size
X = pad_sequences(X, padding='post')

# Define the RNN model
model = Sequential()
model.add(Embedding(input_dim=10, output_dim=8, input_length=5))
model.add(SimpleRNN(16))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model to the data
model.fit(X, y, epochs=10)
