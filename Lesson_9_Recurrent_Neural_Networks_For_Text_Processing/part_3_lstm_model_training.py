# Build the LSTM model
model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=len(dictionary) + 1, output_dim=8, input_length=padded_sentences.shape[1]))
model_lstm.add(LSTM(32))  # Using LSTM instead of SimpleRNN
model_lstm.add(Dense(1, activation='sigmoid'))

model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the LSTM model
model_lstm.fit(X_train, y_train, epochs=10, verbose=1)

# Evaluate the LSTM model
loss_lstm, accuracy_lstm = model_lstm.evaluate(X_test, y_test)
print(f"Test Accuracy (LSTM): {accuracy_lstm}")
