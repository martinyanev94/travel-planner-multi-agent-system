from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample data
# Suppose we have the following embeddings for our words
# For the example, we will use random numbers to simulate embeddings
word_embeddings = {
    'king': np.array([0.1, 0.4, 0.5]),
    'queen': np.array([0.2, 0.4, 0.6]),
    'man': np.array([0.3, 0.1, 0.5]),
    'woman': np.array([0.4, 0.3, 0.7])
}

# Target word
target_word = 'king'

# Preparing the training data
X = []  # features
y = []  # labels

# Creating context words and labels
context_words = ['queen', 'man', 'woman', 'cat']  # Assume 'cat' is irrelevant
for word in context_words:
    X.append(word_embeddings[word])
    # Label is 1 if relevant (appears in the same context as 'king'), 0 otherwise
    y.append(1 if word in ['queen', 'man'] else 0)

# Fitting the model
model = LogisticRegression()
model.fit(X, y)

# Predicting relevance for new context words
test_word_emb = np.array([0.3, 0.4, 0.5])  # New sample embedding
prediction = model.predict([test_word_emb])
print("Prediction for the new context word:", prediction)
