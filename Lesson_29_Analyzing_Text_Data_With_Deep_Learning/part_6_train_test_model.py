from sklearn.model_selection import train_test_split

# Padded sequences
X = pad_sequences(data['vectors'].tolist(), maxlen=input_length)
y = data['labels'].values  # Assuming there is a 'labels' column with the target classes

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
