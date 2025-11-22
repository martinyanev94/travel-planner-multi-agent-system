from transformers import BertForQuestionAnswering

# Load BERT for question-answering
qa_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Assume that the last layer will be replaced for the new task
# In actuality, the code would involve more steps around data preparation and training specifics.
qa_model.classifier = torch.nn.Linear(qa_model.config.hidden_size, num_labels)

# Now the model can be fine-tuned for the new task
