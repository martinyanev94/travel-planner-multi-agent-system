from gensim.models import Word2Vec
import nltk
nltk.download('punkt')

# Your sentences, sure to be a list of tokenized words
sentences = [
    nltk.word_tokenize("Artificial intelligence is one of the greatest technological advancements."),
    nltk.word_tokenize("Natural language processing allows computers to understand human language."),
    nltk.word_tokenize("Text representation is crucial for AI models.")
]

# Training a Word2Vec model
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=4)

# Save the model
model.save("text_representation.model")

# Display the vector for the word "AI"
print(model.wv['Artificial'])
