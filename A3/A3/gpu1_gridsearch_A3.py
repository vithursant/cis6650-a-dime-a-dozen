import os
import pdb


# Grid search for RNN + L2
weight_decay = [1.0, 0.01, 0.001, 0.0001, 0.00004]

for k, weight_decay_factor in enumerate(weight_decay):
	command = 'CUDA_VISIBLE_DEVICES=0 python A3.py ' + \
				' --weight_decay_factor ' + str(weight_decay_factor) + \
						' --max_epoch ' + str(100)
	os.system(command)
