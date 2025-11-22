from tensorflow.keras.callbacks import EarlyStopping

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(train_data.batch(32), validation_data=test_data.batch(32), epochs=10, callbacks=[early_stopping])
test_loss, test_accuracy = model.evaluate(test_data.batch(32))
print(f'Test Accuracy: {test_accuracy:.2f}, Test Loss: {test_loss:.2f}')
