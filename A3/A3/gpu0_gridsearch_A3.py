import os
import pdb

in_keep_prob = [0.25, 0.50, 0.75, 1.0]
out_keep_prob = [0.25, 0.50, 0.75, 1.0]
weight_decay = [1.0, 0.01, 0.001, 0.0001, 0.00004]

for i, in_keep_prob_factor in enumerate(in_keep_prob):
	for j, out_keep_prob_factor in enumerate(out_keep_prob):
		for k, weight_decay_factor in enumerate(weight_decay):
			command = 'CUDA_VISIBLE_DEVICES=0 python A3.py --in_keep_prob ' + str(in_keep_prob_factor) + \
						' --out_keep_prob ' + str(out_keep_prob_factor) + \
							' --weight_decay_factor ' + str(weight_decay_factor) + \
								' --max_epoch ' + str(50)
			os.system(command)
