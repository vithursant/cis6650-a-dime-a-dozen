def gen_imperfect_train_data():
    '''
    Function that generates imperfect data for training input.
    10 samples of perfect data and 10 random samples with randomly
    added noise.
    
    Returns:
        imperfect_input_data:   A numpy array of perfect and noisy input
                                training data
        imperfect_output_data:  A numpy array of corresponding labels
                                for imperfect_input_data
    '''
    imperfect_input_data = []
    imperfect_output_data = []
    
    perfect_input = [ image2bits( image[digit] ) for digit in range(0,10) ];
    perfect_output = [ onehot(digit,10) for digit in range(0,10) ];
    selected_idx = [random.randrange(0,10) for _ in range (0, 10)]
    random_input = [image2bits( image[digit] ) for digit in selected_idx ];
    random_output = [ onehot(digit,10) for digit in selected_idx ];
    
    imperfect_input_data =  perfect_input.append(random_input)
    imperfect_output_data = perfect_output.append(random_output)
    
    return np.vstack(imperfect_input_data).astype(DTYPE), np.vstack(imperfect_output_data).astype(DTYPE)