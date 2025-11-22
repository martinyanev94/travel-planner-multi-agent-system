from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout

# Parameters
embedding_dim = 50
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 for padding

# Building the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
