!pip install gensim
import gensim
from gensim.models import Word2Vec
import numpy as np

# Sample sentences for demonstration
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "lay", "on", "the", "rug"],
    ["the", "cat", "chased", "the", "dog"],
    ["the", "dog", "barked", "at", "the", "cat"],
]
