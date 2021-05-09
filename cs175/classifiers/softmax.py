import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  s = np.zeros((N, C))

  for i in range(N):
    s[i] = X[i].dot(W) # 1 x C
    s_exp = np.exp(s[i]) # 1 x C without divided by sum
    s_exp_sum = s_exp.sum() # y_hat = s_exp / s_exp_sum

    # expand the formula in lecture,
    # we get loss = log(s_exp_sum) - s_y
    loss += np.log(s_exp_sum) - s[i][y[i]]
    
    dW[:, y[i]] -= X[i]
    for j in range(C):
      dW[:, j] += s_exp[j] / s_exp_sum * X[i]
  
  # regularization
  loss = loss / N + 0.5 * reg * (W * W).sum()
  dW = dW / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  # breakpoint()
  s = X.dot(W) # N x C
  s_exp = np.exp(s) # N x C
  s_exp_sum = s_exp.sum(axis=1) # N x 1 same as 1 X N
  loss = np.log(s_exp_sum).sum() - s[range(N), y].sum()
  
  # dW should be D x C
  y_hat = s_exp / s_exp_sum.reshape(-1,1) # N x C # s_exp_sum should be
  # broadcast
  y_hat[range(N), y] -= 1
  dW = X.T.dot(y_hat)
  
  # regularization
  loss = loss / N + 0.5 * reg * (W * W).sum()
  dW = dW / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

