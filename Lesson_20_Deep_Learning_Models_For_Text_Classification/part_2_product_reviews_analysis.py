data = {
    'review': [
        "I love this product! It works great.",
        "This is the worst purchase I have ever made.",
        "Absolutely fantastic! Highly recommend it.",
        "Not worth the money. I'm very disappointed.",
        "This product exceeded my expectations!",
        "Terrible, I will never buy this again."
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
}
df = pd.DataFrame(data)
X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
