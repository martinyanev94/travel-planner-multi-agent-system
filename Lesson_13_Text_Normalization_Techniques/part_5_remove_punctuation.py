import re

# Removing punctuation
cleaned_text = ' '.join(lowercase_tokens)
cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
print(cleaned_text.split())
