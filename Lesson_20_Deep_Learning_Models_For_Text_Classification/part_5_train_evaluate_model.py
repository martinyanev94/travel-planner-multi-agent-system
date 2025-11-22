model.fit(X_train_tfidf.toarray(), y_train, epochs=10, batch_size=4, validation_data=(X_test_tfidf.toarray(), y_test))
loss, accuracy = model.evaluate(X_test_tfidf.toarray(), y_test)
print(f"Test Accuracy: {accuracy:.4f}")
