import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten

# Sample labeled data
texts = [
    "I love programming",
    "This code is bad",
    "Natural language processing is awesome",
    "I hate bugs",
]
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Prepare input vectors using our trained embeddings
input_vectors = np.array([model.wv[text.split()] for text in texts])

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(input_vectors, labels, test_size=0.25, random_state=42)

# Building a simple model
model = Sequential()
model.add(Embedding(input_dim=2000, output_dim=50, input_length=50))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=2)
