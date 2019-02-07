import numpy as np # for matrix maths 

# sigmoid function for two classes
def sigmoid(L):
    for i, number in enumerate(L):
        L[i] = np.divide(1, 1+np.exp(number))
    return L

# softmax function for more than one class 
def softmax(L):
    base_dividend = np.sum(np.exp(L))
    for i, number in enumerate(L):
        L[i] = np.divide(np.exp(number), base_dividend)

    return L


print(sigmoid([0.6, 0.9]))
print(softmax([1,2,3,4]))