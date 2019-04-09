import numpy as np

def grad_descent(xs, ys, alpha=0.1, iters=40):
    (rows, columns) = xs.shape
    theta = np.zeros((columns, 1))
    
    J_history = np.array([])
    
    for _ in range(iters):
        hypothesis = np.dot(xs, theta)
        grad = np.dot(np.transpose(xs), hypothesis - ys)
        theta = theta - (alpha/rows) * grad
        
        J_history = np.append(J_history, cost(xs, ys, theta))
        
    return theta, J_history

def cost(xs, ys, theta):
    m = len(xs)
    
    hypothesis = np.dot(xs, theta)
    differences = np.square(hypothesis - ys)
    return sum(differences)/m