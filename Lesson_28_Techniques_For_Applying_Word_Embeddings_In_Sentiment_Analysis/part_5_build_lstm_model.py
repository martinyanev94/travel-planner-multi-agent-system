from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

def build_model(embedding_matrix, input_length):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], 
                        output_dim=embedding_matrix.shape[1], 
                        weights=[embedding_matrix], 
                        input_length=input_length, 
                        trainable=False))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Example usage
model = build_model(embedding_matrix, input_length=embedding_matrix.shape[0])
model.summary()
