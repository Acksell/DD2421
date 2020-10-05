import random
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Returns 1 or -1 depending on which class the datapoint belongs to.
def target(datapoint):
    return 0

# A linear kernel function 
def linear_kernel(xi, xj):
    return np.dot(xi, xj)

# A linear kernel function 
def get_poly_kernel(order):
    def poly_kernel(xi, xj):
        return (np.dot(xi, xj) + 1)**order
    return poly_kernel

def get_radial_kernel(sigma):
    def radial_kernel(xi, xj):
        return np.exp(-np.linalg.norm(xi -xj)**2/(2*sigma**2))
    return radial_kernel

# precomputes a matrix P which will be reused a lot of times in the objective function.
def precompute_P(dataset, targets, kernel):
    P = np.zeros((N,N))
    for i, xi in enumerate(dataset):
        for j, xj in enumerate(dataset):
            P[i][j] = targets[i]*targets[j]*kernel(xi,xj)
    return P

# Returns the objective function given a kernel and dataset.
def getObjectiveFunction(dataset, targets, kernel):
    P = precompute_P(dataset, targets, kernel) 

    # objective is the dual formulation expression we want to minimize.
    def objective(alphas):
        doublesum = alphas.dot(P).dot(alphas)
        return doublesum/2 - sum(alphas)
    return objective

# returns the value which should be zero under the equality constraint.
def getZerofun(targets):
    def zerofun(alphas):
        return np.dot(alphas, targets)
    return zerofun

def getOffset(alphas, dataset, targets, kernel):
    # TODO: get the support vectors from the nonzero alpha values
    support_alpha = alphas[alphas > 1e-9][0]
    support_index = np.where(alphas == support_alpha)
    support_vector = dataset[support_index]
    print("support_vector: ", support_vector)
    b = 0
    for i, alpha in enumerate(alphas):
        xi = dataset[i]
        b += alpha*targets[i]*kernel(support_vector, xi)
    return b - targets[support_index]

def getIndicatorFunction(alphas, dataset, targets, kernel):
    b = getOffset(alphas, dataset, targets, kernel)
    def indicator(point):
        indicator_sum = 0 
        for i, alpha in enumerate(alphas):
            xi = dataset[i]
            indicator_sum += alpha*targets[i]*kernel(point, xi)
        return indicator_sum - b
    return indicator

if __name__ == "__main__":
    # reorders samples
    np.random.seed(100)
    random.seed(100)

    #### Generate dataset
    classA = np.concatenate(
      (np.random.randn(10, 2)*0.2 + [1.5, 0.5], np.random.randn(10, 2)*0.2 + [-1.5, 0.5])
    )
    classB = np.random.randn(20, 2)*0.2 + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))

    # 1 for class A, -1 for class B
    targets = np.concatenate(
      (np.ones(classA.shape[0]), -np.ones(classB.shape[0]))
    )

    N = inputs.shape[0] # Number of rows ( samples )
    permute=list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    #### Plot the data set and classes.
    plt.plot(
      [p[0] for p in classA],
      [p[1] for p in classA],
      'b.'
    )
    plt.plot(
      [p[0] for p in classB],
      [p[1] for p in classB],
      'r.'
    )
    plt.axis('equal') # Force same scale on both axes

    #### Set up and solve optimisation problem.
    C=None
    bounds=[(0, C) for b in range(N)]  # makes sure that 0 <= alpha_i <= C. 
    start = np.zeros(N)  # start guess of alpha vector.

    # alphas.dot(targets) == 0
    constraint={'type':'eq', 'fun':getZerofun(targets)}

    # kernel = linear_kernel
    # kernel = get_poly_kernel(3)
    kernel = get_radial_kernel(1)

    ret = minimize(getObjectiveFunction(inputs, targets, kernel), start, bounds=bounds, constraints=constraint)
    if ret['success']:
        print("Found solution")
        alphas = ret['x']
        print(alphas)
        print(getOffset(alphas, inputs, targets, kernel))
        #### Plot decision boundary
        xgrid = np.linspace(-5, 5)
        ygrid = np.linspace(-4, 4)
        indicator = getIndicatorFunction(alphas, inputs, targets, kernel)
        grid = np.array([[indicator(np.array([x,y])) for x in xgrid] for y in ygrid]).reshape((len(xgrid),len(ygrid)))
        
        print(xgrid.shape, ygrid.shape, grid.shape)
        # outputs : (50,) (50,) (50, 50)
        
        plt.contour(xgrid, ygrid, grid,
          (-1.0, 0.0, 1.0),
          colors=('red', 'black', 'blue'),
          linewidths=(1 , 3 , 1)
        )
        plt.savefig('svmplot.pdf') # Save a copy in a file
        plt.show()
    else:
        print("Failed to find a solution.")
