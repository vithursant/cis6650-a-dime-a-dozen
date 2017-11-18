import os
import pdb


# Grid search for RNN + Dropout
in_keep_prob = [0.25, 0.50, 0.75, 1.0]
out_keep_prob = [0.25, 0.50, 0.75, 1.0]

for i, in_keep_prob_factor in enumerate(in_keep_prob):
	for j, out_keep_prob_factor in enumerate(out_keep_prob):
		command = 'CUDA_VISIBLE_DEVICES=0 python A3.py --in_keep_prob ' + str(in_keep_prob_factor) + \
					' --out_keep_prob ' + str(out_keep_prob_factor) + \
							' --max_epoch ' + str(50)
		os.system(command)
