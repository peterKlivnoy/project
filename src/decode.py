import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

letter_array = np.loadtxt('data/alphabet.csv', dtype='str', delimiter=',')
probabilities = np.loadtxt('data/letter_probabilities.csv', delimiter=',')
char_to_idx = { str(value) : index for  index, value in enumerate(letter_array)}
idx_to_char = { index: str(value)  for  index, value in enumerate(letter_array)}
matrix_array = np.loadtxt('data/letter_transition_matrix.csv', delimiter=',')

def decode(ciphertext: str, has_breakpoint: bool) -> str:

    

    best_f, _, _, _  = MCMC(ciphertext, 10_000)

    best_f_inverse = f_inv(ciphertext, best_f)

    plaintext = "".join(idx_to_char[idx] for idx in best_f_inverse)

    return plaintext
    
def f_inv( y:str , f: list) -> list:
    '''
    Should return array of x_i given by the current estimate of f
    '''
    f_inverse = [0 for i in range(len(f))]
    for i in range(len(f)):
        f_inverse[f[i]] = i

    x = [f_inverse[char_to_idx[y[j]]] for j in range(len(y))]
    return x

        
def log_likelihood(y: str, f: list[int]) -> float:
    x = f_inv(y, f)

    # Use log of the initial probability
    # Add a tiny epsilon (like 1e-10) to prevent math.log(0) if a prob is exactly 0
    log_prob = math.log(probabilities[x[0]] + 1e-10) 

    for i in range(len(x)-1):
        transition_prob = matrix_array[x[i+1], x[i]]
        log_prob += math.log(transition_prob + 1e-10)

    return float(log_prob)
    

def ratio_calc(current: str, correct:str):
    s = 0
    for i in range(len(correct)):
        if current[i] == correct[i]:
            s+=1
    return s/len(current)
 
def MCMC(y,N_s, correct=None):
    n = 0
    f_initial: list[int] = np.random.permutation(28).tolist()
    f_current = f_initial
    D: defaultdict[tuple[int,...], int] = defaultdict(int)
    bits_per_symbol_history = []
    length = len(y)
    # NEW: Array to track 1 (accepted) or 0 (rejected)
    acceptances = np.zeros(N_s) 
    
    if correct is not None:
        ratios = np.zeros(N_s) 
    else:
        ratios = None
    
    while n < N_s:
        i, j = np.random.choice(28, size=2, replace=False)
        f_proposal = list(f_current)

        f_proposal[i], f_proposal[j] = f_proposal[j], f_proposal[i]
        
        logalpha = min(0, log_likelihood(y, f_proposal) - log_likelihood(y, f_current))
        u = np.random.uniform(0, 1)
        
        if np.log(u) < logalpha:
            f_current = f_proposal
            acceptances[n] = 1  # NEW: Record that we accepted!
            
        D[tuple(f_current)] += 1
        
        
        if correct is not None:
            curr_f = f_inv(y, f_current)
            currtext = "".join(idx_to_char[idx] for idx in curr_f)
            ratios[n] = ratio_calc(current=currtext, correct=correct)
        

        log2_lik = log_likelihood(y, f_current) / np.log(2) 
        bits_per_symbol = -log2_lik / length
        
        bits_per_symbol_history.append(bits_per_symbol)

        n += 1

    best_f = list(max(D, key=lambda k: D[k]))
    
    # NEW: Return the acceptances array too
    return best_f, acceptances, ratios, bits_per_symbol_history
            
    
def compute_markov_entropy(P, M):
    """
    Computes the theoretical entropy rate of the Markov chain in bits per symbol.
    P: 1D numpy array of probabilities (length m)
    M: 2D numpy array where M[i, j] is the transition from j to i
    """
    entropy = 0.0
    num_states = len(P)
    
    # Iterate over all possible previous states (j)
    for j in range(num_states):
        # Iterate over all possible current states (i)
        for i in range(num_states):
            prob_transition = M[i, j]
            
            # We only calculate if probability > 0 to avoid log2(0) errors
            if prob_transition > 0:
                entropy -= P[j] * prob_transition * np.log2(prob_transition)
                
    return entropy


with open('data/sample/ciphertext.txt', 'r', encoding='utf-8') as file:
    # Read the entire contents of the text file into a single string
    ciphertext = file.read()

with open('data/sample/plaintext.txt', 'r', encoding='utf-8') as file:
    # Read the entire contents of the text file into a single string
    correct_text = file.read()

#plaintext = decode(ciphertext, has_breakpoint=False)
print("\nDecoded plaintext:")



# print(len(ciphertext))
# N_s = 3_000
# T = 100 # Set your sliding window length


# lens = [50, 200, 500, 4500]

# # Dictionary to hold the combined 4 arrays for each length
# # Example: all_ratios[50] will eventually be a list of 4 arrays.
# all_ratios = {l: [] for l in lens}  

# for length in lens:
#     print(f"Running MCMC for length: {length}")
    
#     for i in range(4):
#         print(f"  > Trial {i+1}/4")
        
#         # Grab a chunk of text. 
#         # Using `length * i` guarantees the chunks don't overlap!
#         start_idx = 50 * i
#         cipher_chunk = ciphertext[start_idx : start_idx + length]
#         correct_chunk = correct_text[start_idx : start_idx + length]
        
#         # Run the MCMC
#         best_f, acceptances, ratios, bits_per_symbol_history = MCMC(cipher_chunk, N_s, correct_chunk)
        
#         # Save this specific ratio curve to our dictionary
#         all_ratios[length].append(ratios)


# iterations = np.arange(N_s)
# for length in lens:
#     # Stack the 4 ratio arrays on top of each other and average them downwards (axis=0)
#     # This gives us one perfectly averaged array of 10,000 steps
#     avg_ratio = np.mean(all_ratios[length], axis=0)
    
#     # Plot the average line for this length
#     plt.plot(iterations, avg_ratio, label=f"Length: {length}", alpha=0.8)

# plt.title("Average MCMC Accuracy Ratio over Time by Ciphertext Length")
# plt.xlabel("Iteration (t)")
# plt.ylabel("Average Ratio (Accuracy)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# Call the updated function

N_s = 3_000

best_f, acceptances, ratios, bits_per_symbol_history = MCMC(ciphertext, N_s, correct_text)
theoretical_entropy = compute_markov_entropy(probabilities, matrix_array)

print(f"Estimated bits per symbol: {bits_per_symbol_history[-1]:.4f}")
print(f"Theoretical bits per symbol: {theoretical_entropy:.4f}")

iterations = np.arange(N_s)
# Plotting
plt.figure(figsize=(10, 6))

# Plot the MCMC trace
plt.plot(iterations, bits_per_symbol_history, color='blue', alpha=0.8, label='MCMC Estimate')

# Add the theoretical baseline (horizontal red dashed line)
plt.axhline(y=theoretical_entropy, color='red', linestyle='--', linewidth=2, label=f'Theoretical ({theoretical_entropy:.3f} bits)')

plt.title(f"MCMC bits per symbol over time")
plt.xlabel("Iteration (t)")
plt.ylabel("Bits per Symbol")
plt.legend() # Shows the labels we added above!
plt.tight_layout()
plt.show()


"""
# Calculate the moving average. 
# np.ones(T)/T creates a window of weights (e.g. 1/T) that slides across the acceptances array.
# mode='valid' tells numpy to only calculate the average where the window fully overlaps the data (starting at iteration T).
sliding_acceptance_rate = np.convolve(acceptances, np.ones(T)/T, mode='valid')

# The X-axis should start at T and end at N_s
iterations = np.arange(T, N_s + 1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(iterations, sliding_acceptance_rate, color='blue', alpha=0.8)
plt.title(f"MCMC Acceptance Rate over time (Sliding Window T={T})")
plt.xlabel("Iteration (t)")
plt.ylabel("Acceptance Rate")
plt.tight_layout()
plt.show()
"""










# iterations = np.arange(N_s)
# plt.figure(figsize=(10, 6))
# plt.plot(iterations, ratios, color='blue', alpha=0.8)
# plt.title(f"MCMC ratios over time ")
# plt.xlabel("Iteration (t)")
# plt.ylabel("Ratio")
# plt.tight_layout()
# plt.show()



