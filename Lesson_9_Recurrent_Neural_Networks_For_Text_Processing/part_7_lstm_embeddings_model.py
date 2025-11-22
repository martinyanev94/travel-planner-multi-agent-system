from tensorflow.keras.preprocessing.text import Tokenizer

# Assuming pre-trained embeddings are loaded into a dictionary `embedding_matrix`
embedding_matrix = # Load your pre-trained embeddings here

# Build model with pre-trained embeddings
model_with_embeddings = Sequential()
model_with_embeddings.add(Embedding(input_dim=len(dictionary) + 1, output_dim=8, 
                                     weights=[embedding_matrix], input_length=padded_sentences.shape[1],
                                     trainable=False))  # Freeze embeddings
model_with_embeddings.add(LSTM(32))
model_with_embeddings.add(Dense(1, activation='sigmoid'))

model_with_embeddings.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_with_embeddings.fit(X_train, y_train, epochs=10, verbose=1)

loss_embeddings, accuracy_embeddings = model_with_embeddings.evaluate(X_test, y_test)
print(f"Test Accuracy (with embeddings): {accuracy_embeddings}")
