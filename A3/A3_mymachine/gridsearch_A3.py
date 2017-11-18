import os
import pdb

# early_stopping = [1, 5, 10, 15, 20]
# keep_prob = [0.25, 0.50, 0.75, 1.0]
# weight_decay = [1.0, 0.01, 0.001, 0.0001, 0.0004, 0.00004]
early_stopping = [1]
keep_prob = [0.25]
weight_decay = [1.0]

for i, early_stopping_factor in enumerate(early_stopping):
	for j, keep_prob_factor in enumerate(keep_prob):
		for k, weight_decay_factor in enumerate(weight_decay):
			command = 'CUDA_VISIBLE_DEVICES=0 python A3_earlystopping_L2.py --early_stopping_step ' + str(early_stopping_factor) + \
						' --keep_prob ' + str(keep_prob_factor) + \
							' --weight_decay ' + str(weight_decay_factor) + \
								' --max_epoch ' + str(60)
			os.system(command)
