labels = [1, 1, 0, 0, 1]  # Dummy labels for our samples

# Now we train the model
model.fit(embedding_matrix.reshape(1, -1, embedding_matrix.shape[1]), 
          np.array(labels), 
          epochs=10, 
          batch_size=1)
