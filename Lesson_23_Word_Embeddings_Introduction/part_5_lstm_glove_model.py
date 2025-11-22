import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D

# Load GloVe embeddings
embedding_matrix = ...  # Assume this is pre-loaded GloVe embeddings

# Model definition
model = tf.keras.Sequential([
    Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], 
              weights=[embedding_matrix], trainable=False),
    LSTM(32, return_sequences=True),
    GlobalAveragePooling1D(),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
