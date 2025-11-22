from transformers import pipeline

# Load the NER pipeline
ner_pipeline = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')

# Process an example sentence
sentence = "I went to the bank to deposit money."
ner_results = ner_pipeline(sentence)

# Display the identified entities
for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}")
