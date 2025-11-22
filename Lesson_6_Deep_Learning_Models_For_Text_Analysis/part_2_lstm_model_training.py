from keras.layers import LSTM

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=10, output_dim=8, input_length=5))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model to the data
model.fit(X, y, epochs=10)
