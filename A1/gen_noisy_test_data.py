import random

noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30] # List of noise levels

def gen_noisy_test_data():
    '''
    Function to generate 7 sets of test data containing 20 random digits
    from 0 to 9, with noise levels of 0%, 5%, 10%, 15%, 20%, 25%, 30%, 
    respectively.
    
    Returns:
        correct_labels_list: A numpy array of correct targets for the 
                             test.
        test_input_data:     A numpy array of arrays containing randomly 
                             generated digits with noise
    '''
    correct_labels_list = [] # List of correct labels
    test_input_data = [] # List of test input data
    
    for i, noise in enumerate(noise_levels):
        # Random generate of 20 digits and save into correct_labels_list
        correct_labels = [random.randrange(0,10) for _ in range (0, 20)]
        correct_labels_list.append(correct_labels)
        
        # Add noise to the random digits in correct_labels
        test_data = [ addnoise( image2bits( image[digit] ), noise ) for digit in correct_labels] ;
        test_input_data.append(test_data)
    
    return np.vstack(correct_labels_list), np.vstack(test_input_data)