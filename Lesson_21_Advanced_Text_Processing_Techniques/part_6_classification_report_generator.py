from sklearn.metrics import classification_report

# Predictions on the test set
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)  # Convert probabilities to binary outcomes

print(classification_report(y_test, predicted_labels))
