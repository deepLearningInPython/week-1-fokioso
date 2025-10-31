import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.

# Task 1:
# Instructions:
#Write a function that takes one numeric argument as input.
#If the number is larger than zero, the function should return 1, otherwise is should return -1.
#The name of the function should be step

# Your code here:
# -----------------------------------------------

def step(num):
    if num > 0:
        return 1
    else:
        return -1

print(step(9))
# -----------------------------------------------


# Task 2:
# Instructions:
#Write a function that takes in two arguments: a numpy array, and an integer (call argument "cutoff" and set default to 0).
#The function should return a numpy array of the same length, with all elements smaller than the cutoff being set to cutoff).
#The name of the function should be ReLu


# Your code here:
# -----------------------------------------------
def ReLu(array, cutoff = 0):
    array = np.array(array)
    array[array < cutoff] = cutoff
    return array

test_array = np.array([-5, -1, 0, 3, 7])
print(ReLu(test_array))
# -----------------------------------------------


# Task 3:
# Instructions:
#Write a function that takes in a two-dimensional numpy array of size (n, p) and a one-dimensional numpy array of size p.
#The function should start by multiplying the two numpy arrays (matrix multiplication).
#Next, apply the ReLu function from above to the resulting matrix and return the result.
#Name the function neural_net_layer

# Your code here:
# -----------------------------------------------

def neural_net_layer(X, weights, cutoff=0):

    result = np.dot(X, weights)

    
    return ReLu(result, cutoff)

X = np.array([[1, 2, 3],
              [4, 5, 6],
              [-1, -2, -3]])

weights = np.array([0.5, -1, 2])

output = neural_net_layer(X, weights)
print("Neural net layer output:", output)
# ------------------------------------------
