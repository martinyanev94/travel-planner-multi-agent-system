from transformers import pipeline

# Load sentiment-analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

# Making predictions using a pre-trained model
results = sentiment_pipeline(new_texts)

for text, result in zip(new_texts, results):
    print(f"Sentence: '{text}' - Predicted Sentiment: {result['label']} with score: {result['score']:.4f}")
