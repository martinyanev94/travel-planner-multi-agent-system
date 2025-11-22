from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten

# Define the model architecture
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=8, input_length=2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=1)
