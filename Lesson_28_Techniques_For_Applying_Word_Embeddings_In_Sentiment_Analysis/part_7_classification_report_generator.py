from sklearn.metrics import classification_report

# Assume we have some predictions and true values
true_labels = [1, 0, 1, 0, 1]  # These are the real labels
predictions = model.predict(embedding_matrix.reshape(1, -1, embedding_matrix.shape[1]))
predicted_labels = [1 if p > 0.5 else 0 for p in predictions]

print(classification_report(true_labels, predicted_labels))
