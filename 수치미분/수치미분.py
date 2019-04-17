import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np




def Linear_Regression(x,y):
    x = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    w=tf.Variable(tf.random_normal([1]))

    # Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
