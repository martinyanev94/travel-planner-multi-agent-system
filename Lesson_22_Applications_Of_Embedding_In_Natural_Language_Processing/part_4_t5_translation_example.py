from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 model and tokenizer for translation
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Example of English text to translate to French
input_text = "translate English to French: How are you?"

# Tokenizing the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generating translation
output = model.generate(input_ids)
translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(translated_text)
