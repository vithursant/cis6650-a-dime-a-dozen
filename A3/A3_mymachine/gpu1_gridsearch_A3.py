import os
import pdb

#early_stopping = [60]
# in_keep_prob = [0.75]
# out_keep_prob = [0.25, 0.50, 0.75, 1.0]
weight_decay = [1.0, 0.01, 0.001, 0.0001, 0.00004]
# early_stopping = [1]
# keep_prob = [0.25]
# weight_decay = [1.0]

# for i, in_keep_prob_factor in enumerate(in_keep_prob):
# 	for j, out_keep_prob_factor in enumerate(out_keep_prob):
for k, weight_decay_factor in enumerate(weight_decay):
	command = 'python A3_l2.py ' + \
				' --weight_decay_factor ' + str(weight_decay_factor) + \
						' --max_epoch ' + str(50)
	os.system(command)
