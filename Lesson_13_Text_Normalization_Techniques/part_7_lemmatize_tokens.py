from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

lemmatizer = WordNetLemmatizer()
pos_tags = pos_tag(stemmed_tokens)
lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token, tag in pos_tags]
print(lemmatized_tokens)
