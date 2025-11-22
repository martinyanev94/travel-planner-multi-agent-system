from sklearn.utils import class_weight

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', 
                                                  np.unique(true_labels), 
                                                  true_labels)

# Convert class weights to dictionary format for Keras
class_weight_dict = dict(enumerate(class_weights))

# Train the model with class weights
model.fit(embedding_matrix.reshape(1, -1, embedding_matrix.shape[1]), 
          np.array(labels), 
          epochs=10, 
          batch_size=1, 
          class_weight=class_weight_dict)
