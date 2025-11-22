from tensorflow.keras.layers import GRU

# Build the GRU model
model_gru = Sequential()
model_gru.add(Embedding(input_dim=len(dictionary) + 1, output_dim=8, input_length=padded_sentences.shape[1]))
model_gru.add(GRU(32))  # Using GRU instead of LSTM
model_gru.add(Dense(1, activation='sigmoid'))

model_gru.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the GRU model
model_gru.fit(X_train, y_train, epochs=10, verbose=1)

# Evaluate the GRU model
loss_gru, accuracy_gru = model_gru.evaluate(X_test, y_test)
print(f"Test Accuracy (GRU): {accuracy_gru}")
