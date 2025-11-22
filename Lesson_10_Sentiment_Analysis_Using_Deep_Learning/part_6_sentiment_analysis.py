def prepare_review(review):
    processed_review = preprocess_text(review)
    sequence = tokenizer.texts_to_sequences([processed_review])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    return padded_sequence

new_review = "This movie was fantastic! I really loved it."
prepared_review = prepare_review(new_review)

prediction = model.predict(prepared_review)
sentiment = 'Positive' if prediction[0] > 0.5 else 'Negative'
print(f'Sentiment: {sentiment} with probability {prediction[0][0]}')
