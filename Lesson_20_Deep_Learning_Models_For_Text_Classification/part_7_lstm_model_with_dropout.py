from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Embedding(input_dim=X_train_tfidf.shape[1], output_dim=128, input_length=X_train_tfidf.shape[1]))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.5))  # Adding a dropout layer
model.add(Dense(units=1, activation='sigmoid'))
