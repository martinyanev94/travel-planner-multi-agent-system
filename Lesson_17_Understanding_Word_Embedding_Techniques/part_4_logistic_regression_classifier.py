from sklearn.linear_model import LogisticRegression
import numpy as np

# Prepare training data 
X_train =  np.array([model.wv[word] for word in ["cat", "dog"]]).reshape(2, -1)
y_train = np.array([0, 1])  # 0 for cat, 1 for dog

# Initialize and train the model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Test the classifier with a new word
test_word = model.wv["cat"].reshape(1, -1)
print("Prediction for 'cat':", classifier.predict(test_word))
