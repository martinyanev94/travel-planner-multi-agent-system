from sklearn.utils import class_weight

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

# Train the model with class weights
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), class_weight=class_weights)
