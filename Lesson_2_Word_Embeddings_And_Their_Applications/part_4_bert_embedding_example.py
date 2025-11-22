from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input text
input_text = "AI is revolutionizing technology and transforming industries."
inputs = tokenizer(input_text, return_tensors='pt')

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)

# The last hidden states are the output embeddings
last_hidden_states = outputs.last_hidden_state
print("BERT Embedding for the input sentence:", last_hidden_states)
