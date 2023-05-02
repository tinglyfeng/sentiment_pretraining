import numpy as np
import os 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
permutations = np.load('./permutations_hamming_max_1000.npy')
print()