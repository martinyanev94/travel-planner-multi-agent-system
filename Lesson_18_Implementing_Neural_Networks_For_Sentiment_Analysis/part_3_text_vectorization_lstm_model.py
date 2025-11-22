import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

# Create a TextVectorization layer
max_features = 10000  # Number of words to consider
sequence_length = 500  # Maximum length of sentences

text_vectorization = TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=sequence_length)

# Fit the TextVectorization layer on the training data
train_text = train_data.map(lambda x, y: x)  # Extracting text only
text_vectorization.adapt(train_text)
model = tf.keras.Sequential([
    text_vectorization,
    tf.keras.layers.Embedding(input_dim=max_features, output_dim=128),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
