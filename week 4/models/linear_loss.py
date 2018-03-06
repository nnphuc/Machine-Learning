import numpy as np

def linear_loss_naive(W, X, y, reg):
    """
    Linear loss function, naive implementation (with loops)

    Inputs have dimension D, there are N examples.

    Inputs:
    - W: A numpy array of shape (D, 1) containing weights.
    - X: A numpy array of shape (N, D) containing data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c is a real number.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N,D = np.shape(X)
    for i in range(N):
        yML = sum([X[i][j]*W[j] for j in range(D)])
        loss += (y[i]-yML)**2
    
    ww = 0
    for i in range(D):
        ww += W[i]**2
    loss = (loss + reg*ww)/2/N
    dW = 1 / N * np.matmul(np.transpose(X), np.matmul(X, W) - y) + (reg / N) * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def linear_loss_vectorized(W, X, y, reg):
    """
    Linear loss function, vectorized version.

    Inputs and outputs are the same as linear_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = len(X)
    loss = np.dot((np.dot(X,W)-y).T,np.dot(X,W)-y)/(2*N) + np.linalg.norm(W,2)**2 *(reg/2/N)
    dW = np.dot(X.T, np.dot(X, W) - y) / N + (reg/N)*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW