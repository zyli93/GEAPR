import tensorflow as tf

reglr = tf.contrib.layers.l2_regularizer(scale=0.1)
inilz = tf.contrib.layers.xavier_initializer()

with tf.variable_scope("a"):
    x = tf.get_variable("cool", shape=(4,5), dtype=tf.float32,
                        regularizer=reglr, initializer=inilz)

print(x)

with tf.variable_scope("a", reuse=tf.AUTO_REUSE):
    y = tf.get_variable("cool", shape=(4,5), dtype=tf.float32,
                        regularizer=reglr, initializer=inilz)

print(y)

print(x == y)