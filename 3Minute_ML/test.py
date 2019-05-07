import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

total_epoch = 100
batch_size = 10
n_hidden = 256
n_input = 28 * 28
n_noise = 128
n_class = 10

x,y=mnist.train.next_batch(batch_size)
print(y)