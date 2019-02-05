import pandas as pd 

# weights for the first and 
w1_and = 0.5
w2_and = 0.5
b_and  = -1

# weights for the not gate 
w_not = -1.0
b_not = 1.0

# weights for the or gate 
w1_or = 1.0
w2_or = 1.0
b_or  = -1.0

# # weights for the last and gate 
# w1 = 
# w2 = 
# b  = 

# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, True, True, False]
outputs = []

for test_input, correct_output in zip(test_inputs, correct_outputs):
    # test and gate
    linear_and = w1_and * test_input[0] + w2_and * test_input[1] + b_and
    output_and = int(linear_and >=0)
    # not gate after and gate
    linear_not = w_not * output_and + b_not 
    output_not = int(linear_not > 0)
    
    # or gate for the neural network
    linear_or = w1_or * test_input[0] + w2_or * test_input[1] + b_or
    output_or = int(linear_or >=0)

    # last and gate for the final results
    linear_combination = w1_and * output_not + w2_and * output_or + b_and
    output = int(linear_combination >= 0)

    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))
