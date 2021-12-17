def get_conv_output_size(initial_input_size, number_blocks, max_pool=True):
    """ The conv blocks we use keep the same size but use max pooling, so the output of all convolution blocks will be of length input_size / 2"""
    if max_pool==False:
        return initial_input_size
    out_size = initial_input_size
    for i in range(number_blocks):
        out_size = int(out_size / 2)
    return out_size

def get_trans_output_size(input_size, stride, padding, kernel_size):
    """ A function to get the output length of a vector of length input_size after a tranposed convolution layer"""
    return (input_size -1)*stride - 2*padding + (kernel_size - 1) +1

def get_final_output(initial_input_size, number_blocks, number_trans_per_block, stride, padding, kernel_size):
    """A function to get the final output size after tranposed convolution blocks"""
    out_size = initial_input_size
    for i in range(number_blocks):
        for k in range(number_trans_per_block):
            out_size = get_trans_output_size(out_size, stride, padding, kernel_size)
    return out_size
