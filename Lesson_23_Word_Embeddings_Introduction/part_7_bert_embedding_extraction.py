from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode text
inputs = tokenizer("The cat sat on the mat.", return_tensors='pt')

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states)
