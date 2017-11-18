import numpy as np
import pdb
import csv
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import argparse


parser = argparse.ArgumentParser(description='Plot Results')
parser.add_argument('--title', type=str, metavar='N',
                    help='test id number to be used for filenames')
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--both', action='store_true', default=False)
# parser.add_argument('--xmax', type=float, metavar='N',
#                     help='test id number to be used for filenames')
# parser.add_argument('--ymin', type=float, metavar='N',
#                     help='test id number to be used for filenames')
# parser.add_argument('--name', type=str, metavar='N',
#                     help='test id number to be used for filenames')
args = parser.parse_args()

def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

plots = []
for line in open('plots_to_plot.txt'):
  if line.startswith('#'):
    continue
  else:
  	plots.append(line.splitlines()[0])

epochs = init_list_of_objects(len(plots))
train_cost = init_list_of_objects(len(plots))
val_cost = init_list_of_objects(len(plots))

for fnum in range(len(plots)):
  with open(os.getcwd() + plots[fnum],'r') as csvfile:
      variables = csv.reader(csvfile, delimiter=',')
      i = 0
      for row in variables:
        if i >= 9:
          epochs[fnum].append(float(row[0]))
          train_cost[fnum].append(float(row[1]))
          val_cost[fnum].append(float(row[2]))
        i += 1

# labels = ['lambda = 1.0', 
#           'lambda = 0.01', 
#           'lambda = 0.001', 
#           'lambda = 0.0001', 
#           'lambda = 0.00004',
#           'No Regularization']

# 'in_keep_prob = 0.25, out_keep_prob = 0.50', 
# 'in_keep_prob = 0.25, out_keep_prob = 0.75',
# 'in_keep_prob = 0.25, out_keep_prob = 1.00',
# 'in_keep_prob = 0.50, out_keep_prob = 0.25',
# 'in_keep_prob = 0.50, out_keep_prob = 1.00',
# 'in_keep_prob = 0.75, out_keep_prob = 1.00',
# 'in_keep_prob = 1.00, out_keep_prob = 0.25',
# 'in_keep_prob = 1.00, out_keep_prob = 0.50',
# 'in_keep_prob = 1.00, out_keep_prob = 0.75',
# 'in_keep_prob = 1.00, out_keep_prob = 1.00',
# labels = ['in_keep_prob = 0.25, out_keep_prob = 0.25', 
#           'in_keep_prob = 0.50, out_keep_prob = 0.50',
#           'in_keep_prob = 0.50, out_keep_prob = 0.75',
#           'in_keep_prob = 0.75, out_keep_prob = 0.25',
#           'in_keep_prob = 0.75, out_keep_prob = 0.50',
#           'in_keep_prob = 0.75, out_keep_prob = 0.75',
#           'No Regularization']

#labels = ['No Regularization']
# labels = ['in_keep_prob = 0.50, out_keep_prob = 0.50, lambda = 1.0',
#           'in_keep_prob = 0.50, out_keep_prob = 0.50, lambda = 0.01',
#           'in_keep_prob = 0.50, out_keep_prob = 0.50, lambda = 0.001',
#           'in_keep_prob = 0.50, out_keep_prob = 0.50, lambda = 0.0001',
#           'in_keep_prob = 0.50, out_keep_prob = 0.50, lambda = 0.00004',
#           'No Regularization']

labels = ['lambda = 0.00004',
          'in_keep_prob = 0.50, out_keep_prob = 0.50',
          'in_keep_prob = 0.50, out_keep_prob = 0.50, lambda = 0.0001',
          'in_keep_prob = 0.50, out_keep_prob = 0.50, lambda = 0.00004',
          'No Regularization']
maps = ['inferno', 'plasma', 'icefire_r', 'viridis', 'rainbow_r']
cmap = plt.get_cmap(maps[4])
colors = cmap(np.linspace(0, 1, len(labels)))

if args.train:
  for fnum in range(len(plots)-1):
    plt.plot(epochs[fnum][:50], train_cost[fnum][:50], c=colors[fnum], label=labels[fnum])
  #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.plot(epochs[len(plots)-1][:50], train_cost[len(plots)-1][:50], '--', c=colors[len(plots)-1], label=labels[len(plots)-1])
  plt.legend(loc='upper right')
  plt.xlabel("Epochs")
  plt.ylabel("Train Cost")
  plt.xlim()
  plt.ylim()
  plt.title(args.title)
  plt.savefig(args.title + ".pdf", bbox_inches='tight')
  plt.close()

if args.test:
  for fnum in range(len(plots)-1):
    plt.plot(epochs[fnum][:50], val_cost[fnum][:50], c=colors[fnum], label=labels[fnum])
  #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.plot(epochs[len(plots)-1][:50], val_cost[len(plots)-1][:50], '--', c=colors[len(plots)-1], label=labels[len(plots)-1])
  plt.legend(loc='upper right')
  plt.xlabel("Epochs")
  plt.ylabel("Validation Cost")
  plt.xlim()
  plt.ylim()
  plt.title(args.title)
  plt.savefig(args.title + ".pdf", bbox_inches='tight')
  plt.close()