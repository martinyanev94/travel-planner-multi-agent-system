import numpy as np

def load_glove_model(glove_file):
    glove_model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
    return glove_model

# Load GloVe model (make sure the GloVe text file is accessible)
glove_model = load_glove_model("glove.6B.50d.txt")

# Access the embedding for a specific word
print("Embedding for 'technology':", glove_model['technology'])
