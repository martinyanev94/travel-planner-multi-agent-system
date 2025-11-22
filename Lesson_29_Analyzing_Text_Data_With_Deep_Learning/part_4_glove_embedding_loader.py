import numpy as np

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')
def text_to_vector(text, embeddings):
    words = text.split()
    vector = np.mean([embeddings[word] for word in words if word in embeddings], axis=0)
    return vector

data['vectors'] = data['cleaned_reviews'].apply(lambda x: text_to_vector(x, glove_embeddings))
