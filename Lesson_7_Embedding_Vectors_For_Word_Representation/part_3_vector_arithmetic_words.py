# Get vectors for the specified words
king_vector = model.wv['king']
queen_vector = model.wv['queen']
man_vector = model.wv['man']
woman_vector = model.wv['woman']

# Perform vector arithmetic
woman_vector_calculated = king_vector - man_vector + queen_vector
similar_word = model.wv.similar_by_vector(woman_vector_calculated, topn=1)
print(similar_word)  # This should return something similar to 'queen'
