from itertools import combinations
from collections import OrderedDict
import subprocess
import numpy as np

import util

models_2_cmd = OrderedDict({
    # Works
    # 'AIR': [ 
    #     '--model-type', 'AIR',
    #     '--prior_dist', 'Independent',
    #     '--lr', '1e-4', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '3',
    #     '--strokes_per_img', '2',
    #     '--z_where_type', '3',
    #     '--z_what_in_pos', 'z_where_rnn',
    #     '--target_in_pos', 'RNN',
    # ],
    # todo: Doesn't work
    # 'AIR+seq_prior': [
    #     '--model-type', 'AIR',
    #     '--prior_dist', 'Sequential',
    #     '--lr', '1e-4', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '3',
    #     '--strokes_per_img', '2',
    #     '--z_where_type', '3',
    #     '--z_what_in_pos', 'z_where_rnn',
    #     '--target_in_pos', 'RNN',
    # ],
    # todo: Doesn't work
    # 'AIR+seperated_z': [
    #     '--model-type', 'AIR',
    #     '--prior_dist', 'Independent',
    #     '--lr', '1e-4', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '3',
    #     '--strokes_per_img', '2',
    #     '--z_where_type', '3',
    #     '--z_what_in_pos', 'z_what_rnn',
    #     '--target_in_pos', 'RNN',
    # ],
    # Works
    # 'AIR+canvas': [ 
    #     '--model-type', 'AIR',
    #     '--prior_dist', 'Independent',
    #     '--lr', '1e-4', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '3',
    #     '--strokes_per_img', '2',
    #     '--z_where_type', '3',
    #     '--z_what_in_pos', 'z_where_rnn',
    #     '--target_in_pos', 'RNN',
    #     '-eg',
    # ],
    # # todo: collapse
    # 'AIR+spline': [
    #     '--model-type', 'Sequential',     # feature 1 +
    #     '--prior_dist', 'Independent',    # feature 2 -
    #     # '--prior_dist', 'Sequential',    # feature 2 +
    #     '--lr', '1e-3', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '3',
    #     '--strokes_per_img', '2',
    #     # '--z_where_type', '4_rotate',
    #     '--z_where_type', '3',
    #     # '--execution_guided',             # feature 3 +
    #     '--z_what_in_pos', 'z_where_rnn', # feature 4 -
    #     # '--z_what_in_pos', 'z_what_rnn', # feature 4 +
    #     '--target_in_pos', 'RNN',         # feature 5 -
    #     # '--target_in_pos', 'MLP',         # feature 5 +
    # ],
    # 'AIR+spline-spline_decoder=False': [
    #     '--model-type', 'Sequential',     # feature 1 +
    #     '--prior_dist', 'Independent',    # feature 2 -
    #     # '--prior_dist', 'Sequential',    # feature 2 +
    #     '--lr', '1e-4', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '3',
    #     '--strokes_per_img', '2',
    #     # '--z_where_type', '4_rotate',
    #     '--z_where_type', '3',
    #     # '--execution_guided',             # feature 3 +
    #     '--z_what_in_pos', 'z_where_rnn', # feature 4 -
    #     # '--z_what_in_pos', 'z_what_rnn', # feature 4 +
    #     '--target_in_pos', 'RNN',         # feature 5 -
    #     # '--target_in_pos', 'MLP',         # feature 5 +
    #     '--no_spline_renderer',
    # ],
    # kind of working
    # 'AIR+sp+img_std.01-1_tanh-debug': [
    #     '--model-type', 'Sequential',     # feature 1 +
    #     '--prior_dist', 'Independent',    # feature 2 -
    #     '--lr', '1e-3', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '3',
    #     '--strokes_per_img', '2',
    #     '--z_where_type', '3',
    #     '--z_what_in_pos', 'z_where_rnn', # feature 4 -
    #     '--target_in_pos', 'RNN',         # feature 5 -
    #     '--render_method', 'base',
    #     '--no_maxnorm',
    #     '--no_strk_tanh',
    # ],
    # 'AIR-sp-cons_smpl-img_std.01-1_tanh': [
    #     '--model-type', 'Sequential',     # feature 1 +
    #     '--prior_dist', 'Independent',    # feature 2 -
    #     '--lr', '1e-3', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '3',
    #     '--strokes_per_img', '2',
    #     '--z_where_type', '3',
    #     '--z_what_in_pos', 'z_where_rnn', # feature 4 -
    #     '--target_in_pos', 'RNN',         # feature 5 -
    #     '--render_method', 'base',
    #     '--no_maxnorm',
    #     '--no_strk_tanh',
    #     '--constrain_sample',
    # ],
    # Works for at least for 300k iterations
    # 'AIR+spline+_img_std.01-no_zwhere-base_renderer-1_tanh': [
    #     '--model-type', 'Sequential',     # feature 1 +
    #     '--prior_dist', 'Independent',    # feature 2 -
    #     '--lr', '1e-3', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '3',
    #     '--strokes_per_img', '2',
    #     '--z_where_type', '3',
    #     '--z_what_in_pos', 'z_where_rnn', # feature 4 -
    #     '--target_in_pos', 'RNN',         # feature 5 -
    #     '--render_method', 'base',
    #     '--no_maxnorm',
    #     '--no_strk_tanh',
    # ],
    # 'AIR+sp+seq_prir+img_std.01-1_tanh_2': [
    #     '--model-type', 'Sequential',     # feature 1 +
    #     '--prior_dist', 'Sequential',    # feature 2 -
    #     '--lr', '1e-3', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '3',
    #     '--strokes_per_img', '2',
    #     '--z_where_type', '3',
    #     '--z_what_in_pos', 'z_where_rnn', # feature 4 -
    #     '--target_in_pos', 'RNN',         # feature 5 -
    #     '--render_method', 'base',
    #     '--no_maxnorm',
    #     '--no_strk_tanh',
    # ],
    # todo: Nope
    # 'AIR+sp+eg+img_std.01-2_tanh': [
    #     '--model-type', 'Sequential',     # feature 1 +
    #     '--prior_dist', 'Independent',    # feature 2 -
    #     '--lr', '1e-3', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '3',
    #     '--strokes_per_img', '2',
    #     '--z_where_type', '3',
    #     '--execution_guided',
    #     '--z_what_in_pos', 'z_where_rnn', # feature 4 -
    #     '--target_in_pos', 'RNN',         # feature 5 -
    #     '--render_method', 'base',
    #     '--no_maxnorm',
    #     # '--no_strk_tanh',
    # ],
    # 'AIR-img_std.01-2_tanh': [
    #     '--model-type', 'Sequential',     # feature 1 +
    #     '--prior_dist', 'Independent',    # feature 2 -
    #     '--strokes_per_img', '2',
    #     '--lr', '1e-3', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '3',
    #     '--z_what_in_pos', 'z_where_rnn', # feature 4 -
    #     '--target_in_pos', 'RNN',         # feature 5 -
    #     '--render_method', 'base',
    #     '--execution_guided',             # feature 3 +
    #     '--no_maxnorm',
    #     # '--no_strk_tanh', # should use strk_tanh when execution_guided
    #     # '--intermediate_likelihood', 'Mean',
    #     '--intermediate_likelihood', 'Geom',
    # ],
    # 'Full-img_std.01-2tanh-constrain_smpl': [
    #     '--model-type', 'Sequential',     # feature 1 +
    #     '--prior_dist', 'Sequential',    # feature 2 +
    #     '--strokes_per_img', '4',
    #     '--lr', '1e-3', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '3',
    #     '--execution_guided',             # feature 3 +
    #     '--z_what_in_pos', 'z_what_rnn', # feature 4 +
    #     '--target_in_pos', 'MLP',         # feature 5 +
    #     '--render_method', 'base',
    #     '--no_maxnorm',
        # '--constrain_sample',
        # '--no_strk_tanh', # should use strk_tanh when execution_guided
        # '--intermediate_likelihood', 'Mean',
        # '--intermediate_likelihood', 'Geom',
    # ],
    # 'AIR+canvas': [
    #     '--model-type', 'AIR',     # feature 1 -
    #     '--prior_dist', 'Independent',    # feature 2 -
    #     '--strokes_per_img', '4',
    #     '--lr', '1e-4', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '4_rotate',
    #     '--execution_guided',             # feature 3 +
    #     '--z_what_in_pos', 'z_where_rnn', # feature 4 -
    #     '--target_in_pos', 'RNN',         # feature 5 -
    #     # '--render_method', 'base',
    #     '--no_maxnorm',
    #     '--no_strk_tanh',
    # ],'
    # 'Full-sequential_prior-GE': [
    #     '--model-type', 'Sequential',    # feature 1 +
    #     '--prior_dist', 'Independent',    # feature 2 -
    #     '--strokes_per_img', '4',
    #     '--lr', '1e-3', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '4_rotate',
    #                                       # feature 3 -
    # #     '--execution_guided',             # feature 3 +
    #     '--z_what_in_pos', 'z_what_rnn', # feature 4 -
    #     '--target_in_pos', 'MLP',         # feature 5 -
    #     '--render_method', 'base',
    #     '--no_maxnorm',
    #     '--no_strk_tanh',
    # ],
    # 'Full-sequential_prior-GE-sep_z-tar_in_RNN': [
    #     '--model-type', 'Sequential',    # feature 1 +
    #     '--prior_dist', 'Independent',    # feature 2 -
    #     '--strokes_per_img', '4',
    #     '--lr', '1e-3', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '4_rotate',
    #                                       # feature 3 -
    #     '--z_what_in_pos', 'z_where_rnn', # feature 4 -
    #     '--target_in_pos', 'RNN',         # feature 5 -
    #     '--render_method', 'base',
    #     '--no_maxnorm',
    #     '--no_strk_tanh',
    # ],
    # 'AIR+spline_latent_test_again': [
    #     '--model-type', 'Sequential',     # feature 1 +
    #     '--prior_dist', 'Independent',    # feature 2 -
    #     '--strokes_per_img', '4',
    #     '--lr', '1e-3', 
    #     '--bl_lr', '1e-3',
    #     '--z_where_type', '4_rotate',
    #                                       # feature 3 -
    #     # '--execution_guided',
    #     '--z_what_in_pos', 'z_where_rnn', # feature 4 -
    #     '--target_in_pos', 'RNN',         # feature 5 -
    #     '--render_method', 'base',
    #     '--no_maxnorm',
    #     '--no_strk_tanh',
    # ],
})

for n, args in models_2_cmd.items():
    subprocess.run(['python', 'run.py'] + models_2_cmd[n]
                   + ['--save_model_name', n] 
                   + ['--num-iterations', '300000'] 
                #    + ['--continue_training']
                   )