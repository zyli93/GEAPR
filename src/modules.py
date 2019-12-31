"""Modules file, each module, defined as a function,
    is a part of the model.

    @author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

    tf.version: 1.13.1
"""

import tensorflow as tf


def autoencoder(input_features, layers, name_scope, 
        regularizer=None, initializer=None):
    """Auto encoder for structural context of users 

    Args:
        input_features - raw input structural context 
        layers - the structure of enc and dec.
                    [raw dim, hid1_dim, ..., hidk_dim, out_dim]
        scope - name_scope of the ops within the function
        regularizer - the regularizer

    Returns:
        output_feature - the output features
        recon_loss - reconstruction loss 
    """

    with tf.name_scope(name_scope) as scope:

        features = input_features

        # encoder
        for i in range(len(layers) - 1):
            features = tf.layers.dense(inputs=features, units=layers[i+1],
                activation=tf.nn.relu, use_bias=True,
                kernel_regularizer=regularizer, kernel_initializer=initializer,
                bias_regularizer=regularizer, name="usc_enc_{}".format(i))

        # encoded hidden representation
        hidden_feature = feature

        # decoder
        rev_layers = layers[::-1]
        for i in range(1, len(rev_layers) - 2):
            features = tf.layers.dense(inputs=features, units=rev_layers[i+1],
                activation=tf.nn.relu, use_bias=True,
                kernel_regularizer=regularizer, kernel_initializer=initializer,
                bias_regularizer=regularizer, name="usc_dec_{}".format(i))

        # last layer to restore
        restore = tf.layers.dense(inputs=features, units=rev_layers[-1],
                activation=None, use_bias=True,
                kernel_regularizer=regularizer, kernel_initializer=initializer,
                bias_regularizer=regularizer, name="usc_reconstruct_layer")

        # reconstruction loss
        recon_loss = tf.nn.l2_loss(raw_data - restore,
                                   name="recons_loss_{}".format(name_scope))

    return hidden_feature, recon_loss


def centroid(hidden_enc, n_centroid,
             emb_size, tao,
             name_scope, var_name,
             corr_metric,
             regularizer=None,
             activation=None):
    """Model the centroids for users/items

    Centroids mean interests for users and categories for items

    Notations:
        d - embedding_size
        b - batch_size
        c - centroid_size

    Args:
        hidden_enc - the hidden representation of mini-batch matrix, (b,d)
        n_centroid - number of centroids/interests, (c,d)
        emb_size - the embedding size
        tao - [float] the temperature hyper-parameter
        name_scope - the name_scope of the current component
        var_name - the name of the centroid/interest weights
        corr_metric - metrics to regularize the centroids/interests
        activation - [string] of activation functions

    Returns:
        loss - the loss generated from centroid function
    """
    with tf.name_scope(name_scope) as scope:

        # create centroids/interests variables
        with tf.variable_scope(name_scope) as var_scope:
            ctrs = tf.get_variable(shape=[n_centroid, emb_size],
                                   dtype=tf.float32,
                                   name=var_name,
                                   regularizer=regularizer)  # (c,d)

        with tf.name_scope("compute_aggregation") as comp_scope:
            # compute the logits
            outer = tf.matmul(hidden_enc, ctrs, transpose_b=True,
                              name="hemb_ctr_outer")  # (b,c)

            # if `activation` given, pass through activation func
            if activation:
                outer = get_activation_func(activation)\
                    (outer, name="pre_temperature_logits")

            # apply temperature parameter
            outer = outer / tao

            # take softmax
            logits = tf.nn.softmax(outer, axis=-1, name="temperature_softmax")

            # attentional pooling
            output = tf.matmul(hidden_enc, logits, name="attention_pooling")

        with tf.name_scope("correlation_cost") as dist_scope:
            """
                two ways for reduce correlation for centroids:
                    1. Cosine of cosine matrix
                    2. Log of inner product
            """

            # cosine cost
            if corr_metric == "cos":
                numerator = tf.square(tf.matmul(ctrs, ctrs, transpose_b=True))
                row_sqr_sum = tf.reduce_sum(
                    tf.square(ctrs), axis=1, keepdims=True)  # (c,1)
                denominator = tf.matmul(row_sqr_sum, row_sqr_sum, transpose_b=True)
                corr_cost = 0.5 * tf.truediv(numerator, denominator, name="corr_cost_cos")

            # inner product cost
            else:
                mask = tf.ones(shape=(n_centroid, n_centroid), dtype=tf.float32)
                mask -= tf.eye(num_rows=n_centroid, dtype=tf.float32)
                inner = tf.matmul(ctrs, ctrs, transpose_b=True)
                corr_cost = tf.multiply(mask, inner)
                corr_cost = 0.5 * tf.reduce_sum(tf.square(corr_cost), name="corr_cost_log")

            return output, corr_cost


def gatnet():
    """Graph Attention Network component for users/items

    Args:
    """

def mlp(raw_data, layers, name_scope,
        regularizer=None):
    """Multi-layer Perceptron

    :param raw_data:
    :param layers: [raw_dim, layer1, layer2, ...]
    :param name_scope:
    :param regularizer:
    :return:

    TODO:
        - decise whether to use batch normalization
        - decise whether to use dropout
    """

    # implicit community detection
    with tf.name_scope(name_scope):
        for i in range(1, len(layers) - 1):
            feature = tf.layers.dense(feature,
                units=layers[i], activation=tf.nn.relu,
                use_bias=True, kernel_regularizer=regularizer,
                bias_regularizer=regularizer, name="imp_enc_{}".format(i))

        feature = tf.layers.dense(feature,
                units=layers[-1], activation=tf.nn.tanh,
                use_bias=False, kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
                name="imp_enc_{}".format(len(layers)))

        return feature


def get_embedding(inputs, vocab_size,
                  name_scope,
                  num_units, zero_pad=False):
    """Embeds a given tensor.
    """

    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

    return outputs
