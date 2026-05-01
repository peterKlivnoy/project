import math
import time
from collections import defaultdict
import numpy as np

letter_array = np.loadtxt('data/alphabet.csv', dtype='str', delimiter=',')
probabilities = np.loadtxt('data/letter_probabilities.csv', delimiter=',')
log_probabilities = np.log(probabilities + 1e-10)

char_to_idx = { str(value) : index for index, value in enumerate(letter_array)}
idx_to_char = { index: str(value)  for index, value in enumerate(letter_array)}

matrix_array = np.loadtxt('data/letter_transition_matrix.csv', delimiter=',')
log_matrix_array = np.log(matrix_array + 1e-10)
log_prob_3g = np.load('data/3gram_log_matrix.npy')
log_prob_4g = np.load('data/4gram_log_matrix.npy')

def fast_log_likelihood(y_idx_arr, f):
    if len(y_idx_arr) == 0:
        return 0.0
    f_inv_arr = np.zeros(28, dtype=np.int8)
    for index, val in enumerate(f):
        f_inv_arr[val] = index
        
    x = f_inv_arr[y_idx_arr]
    
    log_prob = log_probabilities[x[0]]
    if len(x) > 1:
        log_prob += np.sum(log_matrix_array[x[1:], x[:-1]])
    if len(x) > 2:
        log_prob += np.sum(log_prob_3g[x[:-2], x[1:-1], x[2:]])
    if len(x) > 3:
        log_prob += np.sum(log_prob_4g[x[:-3], x[1:-2], x[2:-1], x[3:]])
        
    return float(log_prob)

def f_inv(y:str , f: list) -> list:
    f_inverse = [0 for i in range(len(f))]
    for i in range(len(f)):
        f_inverse[f[i]] = i
    x = [f_inverse[char_to_idx[y[j]]] for j in range(len(y))]
    return x

def MCMC_with_breakpoint(y: str, max_time=15.0, patience=2500):
    length = len(y)
    y_idx = np.array([char_to_idx[c] for c in y], dtype=np.int8)
    
    f1 = np.random.permutation(28).tolist()
    f2 = np.random.permutation(28).tolist()
    b = np.random.randint(10, length - 10)

    best_log_lik = -float('inf')
    best_state = (list(f1), list(f2), b)
    
    current_log_lik = fast_log_likelihood(y_idx[:b], f1) + fast_log_likelihood(y_idx[b:], f2)
    start_time = time.time()
    iters_since_improvement = 0
    
    while time.time() - start_time < max_time:
        f1_prop, f2_prop, b_prop = list(f1), list(f2), b
        choice = np.random.uniform(0, 1)
        
        if choice < 0.45:
            i, j = np.random.choice(28, size=2, replace=False)
            f1_prop[i], f1_prop[j] = f1_prop[j], f1_prop[i]
        elif choice < 0.90:
            i, j = np.random.choice(28, size=2, replace=False)
            f2_prop[i], f2_prop[j] = f2_prop[j], f2_prop[i]
        elif choice < 0.98:
            step = int(np.random.normal(0, min(15, length * 0.05)))
            b_prop = max(5, min(length - 5, b + step))
        else:
            b_prop = np.random.randint(5, length - 5)
            
        proposal_log_lik = fast_log_likelihood(y_idx[:b_prop], f1_prop) + fast_log_likelihood(y_idx[b_prop:], f2_prop)
        logalpha = min(0, proposal_log_lik - current_log_lik)
        
        if np.log(np.random.uniform(0, 1)) < logalpha:
            f1, f2, b = f1_prop, f2_prop, b_prop
            current_log_lik = proposal_log_lik
            
        if current_log_lik > best_log_lik:
            best_log_lik = current_log_lik
            best_state = (list(f1), list(f2), b)
            iters_since_improvement = 0
        else:
            iters_since_improvement += 1
            
        # Radical Restart to escape local optima
        if iters_since_improvement >= patience:
            f1 = np.random.permutation(28).tolist()
            f2 = np.random.permutation(28).tolist()
            b = np.random.randint(10, length - 10)
            current_log_lik = fast_log_likelihood(y_idx[:b], f1) + fast_log_likelihood(y_idx[b:], f2)
            iters_since_improvement = 0
            
    return best_state

def MCMC(y, max_time=15.0, patience=2000, correct=None):
    length = len(y)
    y_idx = np.array([char_to_idx[c] for c in y], dtype=np.int8)
    
    f_current = np.random.permutation(28).tolist()
    current_log_lik = fast_log_likelihood(y_idx, f_current)
    
    best_log_lik = current_log_lik
    best_f_ever = list(f_current)
    
    start_time = time.time()
    iters_since_improvement = 0
    acceptances = []
    
    while (time.time() - start_time) < max_time:
        i, j = np.random.choice(28, size=2, replace=False)
        f_proposal = list(f_current)
        f_proposal[i], f_proposal[j] = f_proposal[j], f_proposal[i]
        
        proposal_log_lik = fast_log_likelihood(y_idx, f_proposal)
        logalpha = min(0, proposal_log_lik - current_log_lik)
        
        if np.log(np.random.uniform(0, 1)) < logalpha:
            f_current = f_proposal
            current_log_lik = proposal_log_lik
            
        if current_log_lik > best_log_lik:
            best_log_lik = current_log_lik
            best_f_ever = list(f_current)
            iters_since_improvement = 0
        else:
            iters_since_improvement += 1
            
        # Radical Restart to escape local optima
        if iters_since_improvement >= patience:
            f_current = np.random.permutation(28).tolist()
            current_log_lik = fast_log_likelihood(y_idx, f_current)
            iters_since_improvement = 0
            
    return best_f_ever, [], [], []

def decode(ciphertext: str, has_breakpoint: bool) -> str:
    # Set to 110 out of 120s limit just in case
    # This ensures gradescope timeouts DO NOT happen
    if has_breakpoint:
        best_f1, best_f2, best_b = MCMC_with_breakpoint(ciphertext, max_time=20.0)
        
        f1_inverse = f_inv(ciphertext[:best_b], best_f1)
        f2_inverse = f_inv(ciphertext[best_b:], best_f2)
        
        plaintext1 = "".join(idx_to_char[idx] for idx in f1_inverse)
        plaintext2 = "".join(idx_to_char[idx] for idx in f2_inverse)
        
        return plaintext1 + plaintext2
    else:
        best_f, _, _, _  = MCMC(ciphertext, max_time=20.0)

        best_f_inverse = f_inv(ciphertext, best_f)
        plaintext = "".join(idx_to_char[idx] for idx in best_f_inverse)
        
        return plaintext
