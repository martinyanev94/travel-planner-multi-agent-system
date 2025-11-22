from transformers import BertForTokenClassification

# Load the BERT model for token classification
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=9) # Adjust as necessary

# Example sentence
sentence = "Apple Inc. is looking at buying U.K. startup for $1 billion"

# Tokenize the input sentence
inputs = tokenizer(sentence, return_tensors='pt')

# Get predictions
with torch.no_grad():
    outputs = model(**inputs).logits

# Extract token predictions
predictions = torch.argmax(outputs, dim=-1)

# Decode labels
for token, prediction in zip(inputs['input_ids'][0], predictions[0]):
    decoded_token = tokenizer.decode([token])
    print(f'Token: "{decoded_token}" | Predicted label: {prediction.item()}')
