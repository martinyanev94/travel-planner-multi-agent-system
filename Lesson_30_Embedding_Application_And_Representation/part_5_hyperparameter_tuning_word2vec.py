import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Placeholder for results
results = []

# Hyperparameter tuning
vector_sizes = [10, 20, 50]
windows = [2, 3, 5]

for size in vector_sizes:
    for window in windows:
        model = Word2Vec(sentences, vector_size=size, window=window, min_count=1, workers=4)
        df['embeddings'] = df['text'].apply(sentence_to_vector)
        X = list(df['embeddings'])
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((size, window, accuracy))

# Display results to find the best configuration
for result in results:
    print(f"Vector Size: {result[0]}, Window: {result[1]}, Accuracy: {result[2]}")
