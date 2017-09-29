def train_network(epochs):
    sess = tf.Session()   # execute the tensor flow session
    
    
    
    sess.run(tf.global_variables_initializer())   # initialize all the global variables

    train_costs = np.zeros(epochs, dtype='float32');
    train_accuracies = np.zeros(epochs, dtype='float32');
    
    ### TRAINING BEGIN ###
    print ("Epoch  Cost   Accuracy")
    for i in range(epochs):
        input_data, target_data = gen_imperfect_data() # Generates imperfect data every epoch
        train_cost, train_accuracy, _ = sess.run([cost, accuracy, train_op], feed_dict={input_x: input_data, label_y: target_data})
        train_costs[i] = train_cost
        train_accuracies[i] = train_accuracy
        if i % 200 == 0:
            print ("%05d  %5.3f  %5.3f" % (i,train_cost,train_accuracy));
    print ("%05d  %5.3f  %5.3f" % (i,train_cost,train_accuracy));        
    ### TRAINING END ###
    
    return train_costs, train_accuracies, y_pred, sess;