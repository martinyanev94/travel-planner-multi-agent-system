pip install pandas scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the dataset (for illustration, replace 'file_path' with the actual csv file path)
df = pd.read_csv('file_path', encoding='utf-8')

# Define a function to convert sentences to their corresponding embeddings
def sentence_to_vector(sentence):
    words = nltk.word_tokenize(sentence.lower())
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

# Create embeddings for the sentences
df['embeddings'] = df['text'].apply(sentence_to_vector)

# Splitting the dataset into features and labels
X = list(df['embeddings'])
y = df['label']  # assuming the label column contains sentiment labels

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Evaluating the classifier
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
