""" Code for ENGG*6500 Assignment 3
Partly Sunny with a Chance of Hashtags
https://www.kaggle.com/c/crowdflower-weather-twitter
"""
import numpy as np
import cPickle as pickle
import logging

import tensorflow as tf
import os
import time
import csv

import pdb

logger = logging.getLogger(__name__)


### HyperParameters ###
flags = tf.flags
flags.DEFINE_string('data', 'preprocessed_data.pkl', 'path to data file')
flags.DEFINE_string('model', 'model', 'path to model file')
flags.DEFINE_string('optimizer', 'sgd', 'Optimizer to Use')
flags.DEFINE_string('non_linearity', 'tanh', 'Non-Linearity or Activation to use for Hidden Layer')
flags.DEFINE_integer('n_hidden', 50, 'hidden layer size')
flags.DEFINE_integer('n_out', 24, 'output layer size')
flags.DEFINE_integer('n_in', 20, 'input size')
flags.DEFINE_integer('batch_size', 100, 'mini-batch size')
flags.DEFINE_integer('max_epoch', 40, 'maximum number of epochs')
flags.DEFINE_integer('mom_swt', 5, 'Momentum Switch')
flags.DEFINE_integer('save_every', 0, 'Saving frequency(epoch), 0 if you do not wish to save')
flags.DEFINE_integer('val_every', 1, 'Validate frequency(epoch), 0 if you do not wish to validate')
flags.DEFINE_float('lr', 0.01, 'learning rate')
flags.DEFINE_float('lr_decay', 1, 'learning rate decay')
flags.DEFINE_float('init_mom', 0.5, 'initial momentum')
flags.DEFINE_float('final_mom', 0.9, 'final momentum')
flags.DEFINE_integer('early_stopping_step', 10, 'Early stopping step')
flags.DEFINE_float('weight_decay_factor', 0.00004, 'Weight decay')
flags.DEFINE_float('in_keep_prob', 0.5, 'Keep Prob')
flags.DEFINE_float('out_keep_prob', 0.5, 'Keep Prob')

FLAGS = flags.FLAGS

class CSVLogger():
    def __init__(self, filename='log.csv', fieldnames=['epoch']):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in ['early_stopping_step', 'weight_decay_factor', 'in_keep_prob', 'out_keep_prob']:
            writer.writerow([arg, getattr(FLAGS, arg)])
            writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

dataset = 'twitter'

if not os.path.exists('A3_dropout'):
    os.makedirs('A3_dropout')

i = 0
while os.path.isfile('A3_dropout/' + dataset + '_log_' + str(i) + '.csv'):
    i += 1
test_id = i
test_id = str(test_id)

twitter_logger = CSVLogger(filename='A3_dropout/' + dataset + '_log_' + test_id + '.csv', fieldnames=['epochs', 'train_cost', 'val_cost', 'best_cost'])

class RNN(object):

    def __init__(self, max_n_steps, data_x, data_y, raw_n_steps):
        self.n_hidden        = FLAGS.n_hidden
        self.n_out           = FLAGS.n_out
        self.n_in            = FLAGS.n_in
        self.bs              = FLAGS.batch_size
        self.max_epoch       = FLAGS.max_epoch
        self.lr              = FLAGS.lr
        self.init_mom        = FLAGS.init_mom
        self.final_mom       = FLAGS.final_mom
        self.mom_swt         = FLAGS.mom_swt
        self.activation_type = FLAGS.non_linearity
        self.max_n_steps     = max_n_steps
        self.weight_decay_factor    = FLAGS.weight_decay_factor
        params               = self.def_param()

        self.W_o             = params['W_o']
        self.b_o             = params['b_o']
        self.W_h             = params['W_h']
        self.b_h             = params['b_h']
        self.x               = params['in_x']
        self.y               = params['lbl_y']
        self.data_size       = params['data_size']
        self.n_steps         = params['n_steps']

        learned_param        = [self.W_o, self.b_o, self.W_h, self.b_h]
        # get non-linearity
        self.activation      = self.act_fn()

        # output activations
        output_types = [
            (tf.nn.softmax, 0, 5),  # s1-s5
            (tf.nn.softmax, 5, 9),  # w1-w4
            (tf.sigmoid, 9, 24),    # k1-k15
        ]

        # RNN model
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.n_hidden, activation=self.activation)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell, input_keep_prob=FLAGS.in_keep_prob, output_keep_prob=FLAGS.out_keep_prob)

        # define the initial state of the RNN (the hidden input)
        h0 = tf.zeros([tf.shape(self.x)[0], self.n_hidden], dtype ='float32')
        
        # TF cannot perform 3d mult 2d atm so below is a hack
        x_orig_shape = tf.shape(self.x)
        x2d = tf.reshape(self.x,[tf.shape(self.x)[0]*tf.shape(self.x)[1], tf.shape(self.x)[2]]) 
        h_in_rs = tf.matmul(x2d,self.W_h) + self.b_h
        h_in = tf.reshape(h_in_rs,[x_orig_shape[0], x_orig_shape[1], self.n_hidden])

        # inputs has to be [batch_size, max_time, diminput]
        h_os_raw, state = tf.nn.dynamic_rnn(cell=rnn_cell,inputs=h_in,  \
                initial_state=h0, dtype=tf.float32)
        
       
        # NOTE: because recurrent weights are randomly initialized, the zero rows are no longer zero.

        # Gathering only the last outputs of the sequence.
        # TF does not support advanced indexing right now. Below is a hacky way to get around it.
        # we make use of the current batche's longest seq and the reshaping of 3d to 2d to create
        # an index list so tf.gather can be applied to it.
        h_os_raw2d = tf.reshape(h_os_raw, [tf.shape(h_os_raw)[0]*tf.shape(h_os_raw)[1], tf.shape(h_os_raw)[2]])
        n_step_idx = tf.range(0, tf.shape(self.n_steps)[0])*self.max_n_steps + (tf.subtract(self.n_steps,1))
        h_os = tf.gather(h_os_raw2d,n_step_idx, name='Gather_last_out')

        # NOTE: tf.gather_nd's gradient is not supported by TF right now. Advanced indexing will be
        #       much easier with the support using below lines, where n_steps is a 2d map
        # h_os = tf.gather_nd(h_os_raw, self.n_steps)

        if output_types is None:
            self.y_pred = tf.sigmoid(tf.matmul(h_os[-1], self.W_o) + self.b_o)
        else:
            # mixed output types
            # will loop through each type and collect appropriate
            # set of outputs
            ys = []
            for fn, start, end in output_types:
                # (?,38,n_hid) mult (n_hid,n_out)
                ys.append(fn(tf.matmul(h_os,self.W_o[:,start:end]) + self.b_o[start:end]))
                # ys.append(fn(tf.matmul(h_os, self.W_o[:,start:end]) \
                #              + self.by[start:end]))
            self.y_pred = tf.concat(ys, 1)

        # simple mean-squared loss on labels
        self.loss = self.mse()

        # Simple mean-squared loss on labels + l2 weight decay on hidden weights and output weights
        # self.loss = (self.mse() +
        #                 self.weight_decay_factor*tf.nn.l2_loss(self.W_h) +
        #                 self.weight_decay_factor*tf.nn.l2_loss(self.W_o))

    
    def matrix_mult(self, tensor1, tensor2, bias):
        """Multiplies 3d tensor to 2d tensor(matrix)
           return:
             3d original tensor shape.
        """
        t1_shape = tf.shape(tensor1)
        t1_rs = tf.reshape(tensor1,[tf.shape(tensor1)[0]*tf.shape(tensor1)[1], tf.shape(tensor1)[2]])
        out_rs = tf.matmul(t1_rs,tensor2) + bias
        out = tf.reshape(out_rs,t1_shape)

        return out

    def mse(self):
        """ Mean Square Error loss 
        """
        # error between output and target
        E = tf.reduce_mean((self.y_pred - self.y) ** 2)
        
        return E

    def act_fn(self):
        """ choice of non linearity  
        """   

        if self.activation_type == 'sigmoid':
            return tf.sigmoid
        elif self.activation_type == 'tanh':
            return tf.tanh 
        elif self.activation_type == 'relu':
            return tf.nn.relu
        elif self.activation_type == 'cappedrelu':
            return tf.nn.relu6
        else:
            raise NotImplementedError

    def def_param(self):
        """ define parameters required for TF graph.
        """
        #page 44 gradient clipping
        #n initialization from Glorot and Bengio 2010.
        W_hid = tf.Variable(np.random.uniform(low=-0.01, high=0.01,\
                size=(self.n_in, self.n_hidden)), dtype='float32', trainable = True, name='W_h')

        b_hid = tf.Variable(np.zeros([self.n_hidden], dtype='float32'), name='b_h')

        W_out = tf.Variable(np.random.uniform(low=-0.01, high=0.01,\
                size=(self.n_hidden, self.n_out)), dtype='float32',trainable= True,  name='W_o')

        b_out =  tf.Variable(tf.zeros([self.n_out]), name='b_o')

        input_x = tf.placeholder(tf.float32, [None, self.max_n_steps, self.n_in])
        #The label.
        label_y = tf.placeholder(tf.float32, [None,24], name='label_y')

        data_size = tf.placeholder(tf.float32, [None], name= 'data_sze')

        n_steps = tf.placeholder('int32', [None], name='n_steps')

        params = {
            'W_o'       : W_out,
            'b_o'       : b_out,
            'W_h'       : W_hid,
            'b_h'       : b_hid,
            'in_x'      : input_x,
            'lbl_y'     : label_y,
            'data_size' : data_size,
            'n_steps'   : n_steps
            }

        return params


def preproc_input(data_xy, data_max_seq = None,  valid = False):
    """ Preprocesses the data 

    arg:
     data_xy: tuple of our data.
     data_max_seq: maximum sentence size.
     valid: boolean validation data

    """   
   
    data_x, data_y = data_xy

    data_y = np.asarray(data_y)

    if type(data_x) is list:
        n_steps = [x.shape[0] for x in data_x] # get all seq len for each input
        n_steps = np.asarray(n_steps).astype('int32')

        if not valid:
            max_n_steps =max(n_steps)
            # max_n_steps =32 # for debugging test sets
        else:
            max_n_steps = data_max_seq

        n_seq = len(data_x)
        assert(len(data_y) == n_seq)
        n_in = data_x[0].shape[1]
        data_x_p = np.zeros((n_seq, max_n_steps, n_in))

        for i in xrange(n_seq):
            seq_len = data_x[i].shape[0]
            # print data_x[0].shape # (5,20)
            data_x_p[i, :seq_len, :] = data_x[i]

        data_x = data_x_p

    else:
        # n_steps is constant for all sequences
        # this keeps the code consistent for variable and fixed-length
        # sequences as input
        # but we may want to avoid using n_steps altogether
        # when sequences are the same length
        n_steps = x.shape[0] * np.ones((x.shape[1],))
        
        return n_steps

    # preprocess n_steps to feed into the network.
    # n_steps = np.subtract(n_steps ,1)
    # data_size= data_x.shape[1]
    # idx_mask = np.arange(data_size)
    # n_steps = np.column_stack((n_steps, idx_mask))
    if not valid:
        return n_steps, data_x, data_y, max_n_steps
    else:
        return n_steps, data_x, data_y

    
def go_train(model, train_X, train_y, valid_X, valid_y, tr_n_step, vl_n_step):
    """ Trains the built model. 
    arg:
     model: the model to train on.
     train_X: the train data
     train_y: train label data
     valid_X: the validation data
     valid_y: validation label data
     tr_n_step: contains seq length for each train input
     vl_n_step: contains seq length for each validation input

    return: 

    """   
    data_size     = train_X.shape[0]
    max_epoch     = FLAGS.max_epoch
    save_every    = FLAGS.save_every
    val_every     = FLAGS.val_every
    cost_op       = model.loss
    train_op      = tf.train.GradientDescentOptimizer(0.01, name   = 'Gradient_Descent').minimize(cost_op)
    saver         = tf.train.Saver(tf.all_variables(), max_to_keep = 5)
    vl_cost_store = []
    diff          = []

    n_batch_num   = int(np.ceil(1.0 * data_size / model.bs))
    train_costs   = np.zeros(n_batch_num, dtype = 'float32')

    init_var      = tf.initialize_all_variables()
    start_time    = time.clock()
    logger.info("Beginning Training...")
    stopping_step = 0
    with tf.Session() as sess:

        # you need to initialize all variables
        sess.run(init_var)
        best_cost = np.inf
        for i in range(max_epoch):
            n = 0
            for start, end in zip(range(0, data_size, model.bs), range(model.bs, data_size, model.bs)):
                tr_cost, _ = sess.run([cost_op, train_op],feed_dict={model.x: train_X[start:end,:, :], model.y: train_y[start:end, :], model.n_steps: tr_n_step[start:end]})
                train_costs[n] = tr_cost 
                n += 1


            if i % val_every == 0:
                mn_tr_cost_op = tf.reduce_mean(tr_cost)
                vl_cost, mn_tr_cost= sess.run([cost_op,mn_tr_cost_op],\
                        feed_dict={model.x: valid_X, model.y: valid_y, model.n_steps: vl_n_step})

                print ("epoch%d val_cost: %f tr_cost: %f"%(i, vl_cost, mn_tr_cost))

                if vl_cost < best_cost:
                    stopping_step = 0
                    best_cost = vl_cost
                else:
                    stopping_step += 1

                if stopping_step >= FLAGS.early_stopping_step:
                    should_stop = True
                    print("Early stopping is trigger at step: {} val_cost: {}".format(i, vl_cost))
                    terminate(best_cost)

                row = {'epochs': str(i), 'train_cost':str(mn_tr_cost), 'val_cost': str(vl_cost), 'best_cost': str(best_cost)}
                twitter_logger.writerow(row)

                if save_every != 0:
                    if i % save_every == 0:
                        saver.save(sess, 'a3_rnn_model', global_step=i)
             
        terminate(best_cost)
     
    return


def terminate(best_cost):

    print ("Best Found val_cost: %f"%(best_cost))
    logger.info("Train Finished Good Job Computer")

    return


if __name__ == '__main__':
    #data_path = os.path.join('/scratch', os.environ['USER'])
    data_path = os.getcwd()
    #pdb.set_trace()

    logging.basicConfig(level=logging.INFO)
    data_file = os.getcwd() + "/preprocessed_data.pkl"
    #data_file = os.path.join(data_path, "/preprocessed_data.pkl")

    #data_file = FLAGS.data
    if 'train' not in locals() or 'test' not in locals():
        logger.info("loading data from disk...")

        with open(data_file, 'rb') as f:
            train, valid, test = pickle.load(f)

    train_X, train_y = train
    valid_X, valid_y = valid
    # test_X = test

    tr_n_steps, train_X, train_y, max_n_steps = preproc_input((train_X, train_y))
    vl_n_steps, valid_X, valid_y = preproc_input((valid_X, valid_y), max_n_steps, valid=True)

    model = RNN(max_n_steps, train_X, train_y, tr_n_steps)
    go_train(model, train_X, train_y, valid_X, valid_y, tr_n_steps, vl_n_steps)
