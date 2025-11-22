from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences

# Set parameters
max_words = 100
embedding_dim = 100
input_length = max_words  # This example assumes that each input sequence is padded to 100 words

model = Sequential()
model.add(Embedding(input_dim=len(glove_embeddings)+1, output_dim=embedding_dim, input_length=input_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
