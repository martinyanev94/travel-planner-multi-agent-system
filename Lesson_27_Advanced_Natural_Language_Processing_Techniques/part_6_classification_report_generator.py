from sklearn.metrics import classification_report

# Assuming y_true and y_pred are the true labels and predictions
y_true = [1, 0, 1, 1, 0]  # Sample true labels
y_pred = [1, 0, 0, 1, 1]  # Sample predictions

# Generate a classification report
report = classification_report(y_true, y_pred)
print(report)
