import numpy as np
import math
import csv

# Alphabet definitions
alphabet = list("abcdefghijklmnopqrstuvwxyz .")
char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
num_chars = len(alphabet)

# Load Part 1 statistics
letter_array = np.loadtxt('data/alphabet.csv', dtype='str', delimiter=',')
probabilities = np.loadtxt('data/letter_probabilities.csv', delimiter=',')
matrix_array = np.loadtxt('data/letter_transition_matrix.csv', delimiter=',')
# matrix_array[i, j] is the transition from j to i => P(i | j)
# Wait, decode.py uses: transition_prob = matrix_array[x[i+1], x[i]] => P(x[i+1] | x[i])

# Initialize a 3D matrix for 3-gram probabilities. 
# We'll compute P(c1, c2, c3), then save log(P) to avoid repetitive math.log calls.
prob_3g = np.zeros((num_chars, num_chars, num_chars))

# 1. Fill the matrix using Markov approximation from Part 1 parameters
for i, c1 in enumerate(alphabet):
    p1 = probabilities[i]
    for j, c2 in enumerate(alphabet):
        # M[j, i] = transition from c1(i) to c2(j)
        p2_given_1 = matrix_array[j, i]
        for k, c3 in enumerate(alphabet):
            # M[k, j] = transition from c2(j) to c3(k)
            p3_given_2 = matrix_array[k, j]
            
            # P(c1, c2, c3) approx= P(c1) * P(c2|c1) * P(c3|c2)
            prob_3g[i, j, k] = p1 * p2_given_1 * p3_given_2

# 2. Load the 3-gram counts from the downloaded CSV 
# We will replace the A-Z trigram probabilities with the actual frequencies.
counts_3g = {}
total_counts = 0

with open('data/3gram.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row: continue
        text = row[0].lower()
        if len(text) == 3 and all(c in "abcdefghijklmnopqrstuvwxyz" for c in text):
            count = int(row[1])
            counts_3g[text] = count
            total_counts += count

# 3. Overwrite the pure A-Z entries with actual data
# Also apply a small smoothing factor for unseen A-Z trigrams
floor_prob = 0.1 / total_counts if total_counts > 0 else 1e-10

for i, c1 in enumerate("abcdefghijklmnopqrstuvwxyz"):
    for j, c2 in enumerate("abcdefghijklmnopqrstuvwxyz"):
        for k, c3 in enumerate("abcdefghijklmnopqrstuvwxyz"):
            trigram = c1 + c2 + c3
            if trigram in counts_3g:
                # Calculate probability of this specific 3-gram
                # Note: We scale it by the total probability of all A-Z sequences from Part 1
                # Alternatively, just use raw probabilities, but let's use true distribution 
                prob = counts_3g[trigram] / total_counts
            else:
                prob = floor_prob
                
            prob_3g[char_to_idx[c1], char_to_idx[c2], char_to_idx[c3]] = prob

# 4. Convert all probabilities to log-probabilities to save time in MCMC
# Use a tiny epsilon to prevent log(0)
epsilon = 1e-15
log_prob_3g = np.log(prob_3g + epsilon)

# 5. Save the resulting 3D matrix for use in decode.py
np.save("data/3gram_log_matrix.npy", log_prob_3g)

print("Successfully built and saved 3gram_log_matrix.npy!")
print(f"Matrix shape: {log_prob_3g.shape}")
