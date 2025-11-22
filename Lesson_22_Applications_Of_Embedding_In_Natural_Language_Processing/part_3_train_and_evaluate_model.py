model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))
from sklearn.metrics import classification_report

# Predictions
predictions = model.predict(x_test)
predicted_labels = (predictions > 0.5).astype(int)

# Generate a classification report
print(classification_report(y_test, predicted_labels))
