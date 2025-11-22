from transformers import BertForSequenceClassification

# Load the model for classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# You would train the model here on your specific dataset...
