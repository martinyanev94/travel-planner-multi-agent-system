from transformers import BertTokenizer, BertModel
import torch

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example sentence
sentence = "The bank can give you a loan."

# Tokenize the input sentence
inputs = tokenizer(sentence, return_tensors='pt')

# Get the contextual embeddings
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

# Extract the embeddings for the word "bank"
token_ids = inputs['input_ids'][0]
word_index = token_ids.tolist().index(tokenizer.encode("bank", add_special_tokens=False)[0])
bank_embedding = last_hidden_states[0][word_index]

print(f'Embedding for "bank": {bank_embedding}')
