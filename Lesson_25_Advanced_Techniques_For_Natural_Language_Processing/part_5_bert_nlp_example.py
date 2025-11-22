from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "Natural Language Processing allows machines to understand human languages."
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

# Get the last hidden state
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
