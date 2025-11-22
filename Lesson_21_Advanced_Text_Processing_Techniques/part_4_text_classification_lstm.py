from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Sample data: sentences and their corresponding labels
texts = [
    "I love programming!",
    "NLP is fascinating.",
    "Python is great for data science.",
    "I hate bugs in my code."
]
labels = [1, 1, 1, 0]  # 1: positive, 0: negative

# Tokenization and padding
max_length = 10
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=max_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Creating the model with an embedding layer
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=10, verbose=1)

# Evaluating the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")
