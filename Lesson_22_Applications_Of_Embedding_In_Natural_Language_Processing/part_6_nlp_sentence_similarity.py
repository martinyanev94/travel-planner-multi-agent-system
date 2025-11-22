from sentence_transformers import SentenceTransformer, util

# Load pretrained model for generating sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample text paragraphs
text_paragraphs = [
    "Natural language processing (NLP) is a subfield of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language.",
    "Machine learning is a crucial aspect of NLP that allows systems to learn from data and improve over time.",
    "Applications of NLP include sentiment analysis, translation, chatbot development, and text summarization."
]

# Generate embeddings for the text paragraphs
embeddings = model.encode(text_paragraphs)

# Calculate cosine similarities from the document embedding
document_embedding = np.mean(embeddings, axis=0)
similarities = util.cos_sim(document_embedding, embeddings)

# Sort sentences based on similarity scores
sorted_indices = np.argsort(similarities[0])[::-1]
for index in sorted_indices:
    print(f"Sentence: {text_paragraphs[index]}, Score: {similarities[0][index]}")
