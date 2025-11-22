def predict_context_word(input_word, tokenizer):
    word_sequence = tokenizer.texts_to_sequences([input_word])[0]
    word_sequence = pad_sequences([word_sequence], maxlen=2)
    
    predicted_word_index = np.argmax(model.predict(word_sequence), axis=-1)[0]
    predicted_word = tokenizer.index_word[predicted_word_index]
    
    return predicted_word

# Test the function
test_word = "the cat"
predicted_context = predict_context_word(test_word, tokenizer)
print(f"The predicted context word for '{test_word}' is '{predicted_context}'.")
