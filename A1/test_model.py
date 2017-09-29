from __future__ import division

def test_model(test_input_data, one_hot_output_data):
    '''
    Function for testing the trained MLP model using the 
    test_input_data and correct_labels_list from Part A.

    Inputs:
        test_input_data:        A numpy array of arrays containing
                                the images in bits for testing model.
        one_hot_output_data:    A numpy array of arrays containing
                                one-hot encoded corresponding targets
    
    Returns:
        test_accuracy_list:     A numpy array of test accuracies
                                done at all noise levels.
    '''
    test_accuracy_list = []

    for i, noise in enumerate(noise_levels):
        print("Noise Level: {}".format(noise_levels[i]))
        
        # Feed random noisy test_input_data and corresponding one_hot labels
        outputs = sess.run( y_pred, feed_dict={input_x: test_input_data[i], label_y: one_hot_output_data[i]});
        
        predicted_correct = 0 # Correct prediction count
        
        for digit in correct_labels_list[i]:
            print (bits2image( test_input_data[i][digit] ));
            print ([ "%3.1f" % o for o in outputs[digit] ]);

            o = outputs[digit].tolist();
            
            # Print prediction and actual answer
            print ("Prediction: %d" % (o.index(max(o))))
            print ("Actual: %d"% (np.where(np.array(one_hot_output_data[i][digit])==1)[0]))
            
            # Check if classification right or wrong
            if o.index(max(o)) == np.where(np.array(one_hot_output_data[i][digit])==1)[0]:
                predicted_correct += 1
                print ("Correct\n")
            else:
                print("Incorrect\n")
            print;
            print (80*"-");
            
        # Compute the test accuracy for this test set and append to list
        test_accuracy = (predicted_correct/(len(correct_labels_list[i])))*100
        print("Noise Level: {} Accuracy: {}%\n\n".format(noise, test_accuracy))
        test_accuracy_list.append(test_accuracy)

    return np.vstack(test_accuracy_list)