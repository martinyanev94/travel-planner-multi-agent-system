import random

def generate_negative_samples(target_word, num_samples=2):
    # Dummy vocabulary for demonstration
    vocabulary = ['king', 'queen', 'man', 'woman', 'cat', 'dog', 'apple', 'orange']
    negative_samples = []
    for _ in range(num_samples):
        sample = random.choice(vocabulary)
        while sample == target_word:  # Ensure we do not select the target word
            sample = random.choice(vocabulary)
        negative_samples.append(sample)
    return negative_samples

# Generate negative samples for 'king'
negative_samples = generate_negative_samples('king')
print("Negative samples for 'king':", negative_samples)
