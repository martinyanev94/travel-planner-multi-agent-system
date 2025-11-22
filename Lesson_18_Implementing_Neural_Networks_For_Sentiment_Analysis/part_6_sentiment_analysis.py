def predict_sentiment(text):
    # Process the incoming text
    processed_text = text_vectorization(tf.convert_to_tensor([text]))
    prediction = model.predict(processed_text)
    
    return "Positive" if prediction[0][0] > 0.5 else "Negative"

# Test the sentiment prediction
print(predict_sentiment("I absolutely loved this movie!"))
print(predict_sentiment("This is the worst movie I've ever seen."))
