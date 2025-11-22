import numpy as np

# Load GloVe vectors
def load_glove_model(filepath):
    glove_model = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            glove_model[word] = vectors
    return glove_model

glove_model = load_glove_model('glove.6B.100d.txt')  # Update the path accordingly
