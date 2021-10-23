import subprocess
from itertools import product
import random
import pickle

bl_lr = [1e-1, 1e-2, 1e-3]
num_layers = [2, 3, 4, 5]
rnn_hid_dim = [256, 512]
mlp_hid_dim = [256, 512]
maxnorm = [True, False]

param_list = list(product(bl_lr, num_layers, rnn_hid_dim, mlp_hid_dim, maxnorm))
random.shuffle(param_list)

# Load the record
try:
    with open('save/bl_hypero_searched_list.pt', 'rb') as f:
        searched_list = pickle.load(f)
except FileNotFoundError:
    searched_list = []

for lr, n_lyr, rnn_h_dim, mlp_h_dim, use_maxnorm in param_list:
    if [lr, n_lyr, rnn_h_dim, mlp_h_dim, use_maxnorm] in searched_list:
        # if searched, skip
        continue
    else:
        # Update the record
        searched_list.append([lr, n_lyr, rnn_h_dim, mlp_h_dim, use_maxnorm])
        with open('save/bl_hypero_searched_list.pt', 'wb') as f:
            pickle.dump(searched_list, f)

        if use_maxnorm:
            subprocess.run(['python', 'run.py', '--bl_lr', str(lr), 
                                            '--num_baseline_layers', str(n_lyr), 
                                            '--bl_rnn_hid_dim', str(rnn_h_dim),
                                            '--bl_mlp_hid_dim', str(mlp_h_dim),
                                            '-eg'])
        else:
            subprocess.run(['python', 'run.py', '--bl_lr', str(lr), 
                                            '--num_baseline_layers', str(n_lyr), 
                                            '--bl_rnn_hid_dim', str(rnn_h_dim),
                                            '--bl_mlp_hid_dim', str(mlp_h_dim),
                                            '--maxnorm', '-eg'])