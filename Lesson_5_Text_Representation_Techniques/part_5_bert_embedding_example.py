!pip install transformers

from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample sentence
text = "I love programming with transformers."
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

# Getting the embeddings for the [CLS] token
cls_embedding = outputs.last_hidden_state[0][0].detach().numpy()
print("BERT Embedding for the sentence:", cls_embedding)
