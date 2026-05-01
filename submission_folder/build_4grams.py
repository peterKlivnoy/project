import numpy as np
import math
import csv

alphabet = list("abcdefghijklmnopqrstuvwxyz .")
char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
num_chars = len(alphabet)

probabilities = np.loadtxt('data/letter_probabilities.csv', delimiter=',')
matrix_array = np.loadtxt('data/letter_transition_matrix.csv', delimiter=',')

prob_4g = np.zeros((num_chars, num_chars, num_chars, num_chars))

print("Filling 4-gram matrix using Markov approximations...")
for i in range(num_chars):
    p1 = probabilities[i]
    for j in range(num_chars):
        p2 = matrix_array[j, i]
        for k in range(num_chars):
            p3 = matrix_array[k, j]
            for l in range(num_chars):
                p4 = matrix_array[l, k]
                prob_4g[i, j, k, l] = p1 * p2 * p3 * p4

counts_4g = {}
total_counts = 0

print("Loading 4-gram data from CSV...")
with open('data/4gram.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row: continue
        text = row[0].lower()
        if len(text) == 4 and all(c in "abcdefghijklmnopqrstuvwxyz" for c in text):
            count = int(row[1])
            counts_4g[text] = count
            total_counts += count

print("Normalizing and overwriting 4-grams...")
total_pure_prob = 0
for i, c1 in enumerate("abcdefghijklmnopqrstuvwxyz"):
    for j, c2 in enumerate("abcdefghijklmnopqrstuvwxyz"):
        for k, c3 in enumerate("abcdefghijklmnopqrstuvwxyz"):
            for l, c4 in enumerate("abcdefghijklmnopqrstuvwxyz"):
                total_pure_prob += prob_4g[char_to_idx[c1], char_to_idx[c2], char_to_idx[c3], char_to_idx[c4]]

floor_prob = 0.1 / total_counts if total_counts > 0 else 1e-10

for i, c1 in enumerate("abcdefghijklmnopqrstuvwxyz"):
    for j, c2 in enumerate("abcdefghijklmnopqrstuvwxyz"):
        for k, c3 in enumerate("abcdefghijklmnopqrstuvwxyz"):
            for l, c4 in enumerate("abcdefghijklmnopqrstuvwxyz"):
                quadgram = c1 + c2 + c3 + c4
                if quadgram in counts_4g:
                    prob = (counts_4g[quadgram] / total_counts) * total_pure_prob
                else:
                    prob = floor_prob
                    
                prob_4g[char_to_idx[c1], char_to_idx[c2], char_to_idx[c3], char_to_idx[c4]] = prob

epsilon = 1e-15
log_prob_4g = np.log(prob_4g + epsilon)

np.save("data/4gram_log_matrix.npy", log_prob_4g)
print("Successfully built and saved 4gram_log_matrix.npy!")
print(f"Matrix shape: {log_prob_4g.shape}")
