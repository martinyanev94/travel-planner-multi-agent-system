new_reviews = [
    "I had an enjoyable experience with this product.",
    "This was a terrible waste of money."
]

new_reviews_tfidf = tfidf_vectorizer.transform(new_reviews)
predictions = model.predict(new_reviews_tfidf.toarray())

for review, prediction in zip(new_reviews, predictions):
    print(f"Review: '{review}' - Sentiment: {'Positive' if prediction[0] > 0.5 else 'Negative'}")
