!pip install transformers

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Sample text input
text = "Deep learning models are fascinating!"
inputs = tokenizer(text, return_tensors='pt')

# Forward pass to get logits
with torch.no_grad():
    logits = model(**inputs).logits

print("Logits:", logits)
