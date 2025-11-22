pip install transformers
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

text = "Natural language processing allows machines to understand human language."
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
