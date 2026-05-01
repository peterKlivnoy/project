import math
import time
from collections import defaultdict
import numpy as np

letter_array = np.loadtxt('data/alphabet.csv', dtype='str', delimiter=',')
probabilities = np.loadtxt('data/letter_probabilities.csv', delimiter=',')
char_to_idx = { str(value) : index for  index, value in enumerate(letter_array)}
idx_to_char = { index: str(value)  for  index, value in enumerate(letter_array)}
matrix_array = np.loadtxt('data/letter_transition_matrix.csv', delimiter=',')

def decode(ciphertext: str, has_breakpoint: bool) -> str:
    
    # max_time gives 105 seconds to MCMC out of the total 120 second timeout from Gradescope
    best_f, _, _, _  = MCMC(ciphertext, max_time=105.0)

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
 max_time=100.0, correct=None):
    n = 0
    f_initial: list[int] = np.random.permutation(28).tolist()
    f_current = f_initial
    D: defaultdict[tuple[int,...], int] = defaultdict(int)
    bits_per_symbol_history = []
    length = len(y)
    
    # Early stopping trackers
    best_log_lik = -float('inf')
    best_f_ever = f_initial

    # NEW: Array to track 1 (accepted) or 0 (rejected)
    acceptances = [] 
    
    if correct is not None:
        ratios = [] 
    else:
        ratios = None
    
    current_log_lik = log_likelihood(y, f_current)
    
    start_time = time.time()
    
    while (time.time() - start_time) < max_time:
        i, j = np.random.choice(28, size=2, replace=False)
        f_proposal = list(f_current)

        f_proposal[i], f_proposal[j] = f_proposal[j], f_proposal[i]
        
        proposal_log_lik = log_likelihood(y, f_proposal)
        
        logalpha = min(0, proposal_log_lik - current_log_lik)
        u = np.random.uniform(0, 1)
        
        if np.log(u) < logalpha:
            f_current = f_proposal
            current_log_lik = proposal_log_lik
            acceptances.append(1)  # NEW: Record that we accepted!
        else:
            acceptances.append(0)
            
        D[tuple(f_current)] += 1
        
        # Best Configuration Tracking
        if current_log_lik > best_log_lik:
            best_log_lik = current_log_lik
            best_f_ever = list(f_current)nt(f"Early stopping at iteration {n}")
            break
        
        
        if correct is not None:
            curr_f = f_inv(y, f_current)
            currtext = "".join(idx_to_char[idx] for idx in curr_f)
            ratios.append(ratio_calc(current=currtext, correct=correct))
        

        log2_lik = current_log_lik / np.log(2) 
        bits_per_symbol = -log2_lik / length
        
        bits_per_symbol_history.append(bits_per_symbol)

        n += 1

    # best_f = list(max(D, key=lambda k: D[k]))   # Instead of most visited state, use highest max log likelihood encountered 
    best_f = best_f_ever
    
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


