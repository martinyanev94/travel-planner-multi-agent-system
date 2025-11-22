import numpy as np

# Load the GloVe model
def load_glove_model(glove_file):
    glove_model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
    return glove_model

glove_model = load_glove_model("glove.6B.50d.txt")

# Access the embedding for the word "language"
print(glove_model['language'])
