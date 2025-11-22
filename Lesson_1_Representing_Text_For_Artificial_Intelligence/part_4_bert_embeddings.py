from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input text
input_text = "AI is transforming industries and pushing technology forward."
inputs = tokenizer(input_text, return_tensors='pt')

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)

# The last hidden state is the output embeddings
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
