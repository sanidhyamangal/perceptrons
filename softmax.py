import numpy as np # for matrix maths 

def softmax(L):
    base_dividend = np.sum(np.exp(L))
    for i, number in enumerate(L):
        L[i] = np.divide(np.exp(number), base_dividend)

    return L

print(softmax([1,2,3,4]))