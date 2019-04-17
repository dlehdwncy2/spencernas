import tensorflow as tf


a=tf.constant(32)
b=tf.constant(16)


add_=tf.add(a,b)

sess=tf.Session()
sess.run(add_)
