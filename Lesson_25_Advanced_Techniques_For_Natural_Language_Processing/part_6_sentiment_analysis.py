def evaluate_sentiment(text):
    # Assume this function uses a pre-trained sentiment model
    # Let's just return dummy values for the sake of this example
    return "positive" if "love" in text else "negative"

test_sentences = [
    "I love programming.",
    "Programming is difficult.",
    "Everyone should have equal rights."
]

for sentence in test_sentences:
    sentiment = evaluate_sentiment(sentence)
    print(f"Sentence: '{sentence}' - Sentiment: {sentiment}")
