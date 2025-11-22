pip install keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences

# Sample data
sentences = [
    ['this', 'movie', 'is', 'great'],
    ['this', 'movie', 'is', 'awful'],
    ['I', 'love', 'this', 'film'],
    ['I', 'hated', 'this', 'film']
]

labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Create an embedding layer
embedding_layer = Embedding(input_dim=len(model.wv.key_to_index), 
                             output_dim=100, 
                             weights=[model.wv.vectors], 
                             trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Prepare sequences
X = [list(map(lambda x: model.wv.key_to_index[x], sentence)) for sentence in sentences]
X = pad_sequences(X)

# Fit the model
model.fit(X, labels, epochs=10)
