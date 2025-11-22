!pip install transformers torch

from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode some text
input_text = "The cat sat on the mat."
inputs = tokenizer(input_text, return_tensors='pt')

# Forward pass through the model
with torch.no_grad():
    outputs = model(**inputs)

# Output the last hidden state
last_hidden_state = outputs.last_hidden_state
print(last_hidden_state)
