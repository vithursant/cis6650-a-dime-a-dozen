def one_hot_test_target_data(correct_labels_list):
    '''
    Function to one-hot encode the test target digits.
    
    Input:
        correct_labels_list: A list of lists containing targets for each
                             test set.
    
    Returns:
        one_hot_output_data: A list of lists containing one-hot encoded
                             targets.
    '''
    for i in range(len(correct_labels_list)):
        output_data = [ onehot(digit,10) for digit in correct_labels_list[i]]
        one_hot_output_data.append(output_data)
    
    return one_hot_output_data