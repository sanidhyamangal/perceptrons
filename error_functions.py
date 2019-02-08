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

# cross-entropy function 
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)

    entropy = -np.sum((Y) * np.log(P) + (1- Y) * np.log(1 - P))

    return entropy
print(sigmoid([0.6, 0.9]))
print(softmax([1,2,3,4]))
print(cross_entropy([1,1,0], [0.8, 0.7, 0.1]))