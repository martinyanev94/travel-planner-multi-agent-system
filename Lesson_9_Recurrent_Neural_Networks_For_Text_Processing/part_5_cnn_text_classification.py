from tensorflow.keras.layers import Conv1D, MaxPooling1D

# Build the CNN model
model_cnn = Sequential()
model_cnn.add(Embedding(input_dim=len(dictionary) + 1, output_dim=8, input_length=padded_sentences.shape[1]))
model_cnn.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Dense(1, activation='sigmoid'))

model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the CNN model
model_cnn.fit(X_train, y_train, epochs=10, verbose=1)

# Evaluate the CNN model
loss_cnn, accuracy_cnn = model_cnn.evaluate(X_test, y_test)
print(f"Test Accuracy (CNN): {accuracy_cnn}")
