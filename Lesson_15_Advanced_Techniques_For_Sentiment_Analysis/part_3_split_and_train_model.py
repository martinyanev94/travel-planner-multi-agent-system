from sklearn.model_selection import train_test_split

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Fitting the model
model.fit(X_train, y_train, epochs=10, batch_size=2)
