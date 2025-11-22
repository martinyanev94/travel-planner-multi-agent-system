from transformers import BertForSequenceClassification

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Prepare sample text for sentiment analysis
texts = ["I love this product!", "This is the worst service ever!", "It's okay, not great but not bad either."]

# Tokenize inputs
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Get predictions
with torch.no_grad():
    logits = model(**inputs).logits

# Get predicted classes
predictions = torch.argmax(logits, dim=-1)

for text, sentiment in zip(texts, predictions):
    print(f'Text: "{text}" | Predicted sentiment: {sentiment.item()}')
