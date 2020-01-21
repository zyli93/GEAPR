import tensorflow as tf

reglr = tf.contrib.layers.l2_regularizer(scale=0.1)
inilz = tf.contrib.layers.xavier_initializer()

"""
with tf.variable_scope("a"):
    x = tf.get_variable("cool", shape=(4,5), dtype=tf.float32,
                        regularizer=reglr, initializer=inilz)

print(x)

with tf.variable_scope("a", reuse=tf.AUTO_REUSE):
    y = tf.get_variable("cool", shape=(4,5), dtype=tf.float32,
                        regularizer=reglr, initializer=inilz)

print(y)

print(x == y)
"""

with tf.name_scope("ns"):
    na = tf.get_variable("ns_gv", shape=(4,5), dtype=tf.float32,
                         regularizer=reglr, initializer=inilz)
    nb = tf.layers.dense(na, 3, name="ls_layerdense")

with tf.variable_scope("vs"):
    va = tf.get_variable("vs_gv", shape=(4,5), dtype=tf.float32,
                         regularizer=reglr, initializer=inilz)
    vb = tf.layers.dense(nb, 3, name="vs_layerdense")
    with tf.variable_scope("vvs"):
        va = tf.get_variable("vvs_gv", shape=(4, 5), dtype=tf.float32,
                             regularizer=reglr, initializer=inilz)

with tf.variable_scope("vs"):
    va = tf.get_variable("vs_gv_2", shape=(4,5), dtype=tf.float32,
                         regularizer=reglr, initializer=inilz)

l = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "vs")

for ll in l:
    print(ll)
