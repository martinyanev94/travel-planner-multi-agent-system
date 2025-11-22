from sklearn.metrics import classification_report, confusion_matrix

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", confusion)
