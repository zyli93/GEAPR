"""Modules file, each module, defined as a function,
    is a part of the model.

    @author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

    tf.version: 1.13.1

    Notes:
        1. Pay attention to zero padding for every get_embeddings()
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

        # last layer to reconstruct
        restore = tf.layers.dense(inputs=features, units=rev_layers[-1],
                activation=None, use_bias=True,
                kernel_regularizer=regularizer, kernel_initializer=initializer,
                bias_regularizer=regularizer, name="usc_reconstruct_layer")

        # reconstruction loss
        recon_loss = tf.nn.l2_loss(raw_data - restore,
                                   name="recons_loss_{}".format(name_scope))

    return hidden_feature, recon_loss


def attentional_fm(name_scope, input_features, emb_dim, feat_size,
                   initializer=None, regularizer=None, dropout_keep=None):
    """attentional factorization machine for attribute feature extractions

    Shapes:
        b - batch_size
        k - number of fields
        d - embedding_size
        |A| - total number of attributes

    Args:
        name_scope - [str]
        input_features - [int] (b, k) input discrete features
        emb_dim - [int] dimension of each embedding, d
        feat_size - [int] total number of distinct features (fields) for FM, A
        attr_size - [int] total number of fields , abbrev. k
        dropout_keep - [bool] whether to use dropout in AFM

    Returns:
        features - 
        attentions - 

    TODO: what is attribute size

    """

    # TODO: what is count here?

    with tf.variable_scope(name_scope) as scope:
        embedding_mat = get_embeddings(vocab_size=feat_size, num_units=emb_dim,
            name_scope=scope, zero_pad=True)  # (|A|+1, d) lookup table for all attr emb 
        uattr_emb = tf.nn.embedding_lookup(embedding_mat, input_features)  # (b, k, d)
        element_wise_prod_list = []
        count = 0

        attn_W = tf.get_variable(name="attention_W", dtype=tf.float32,
            shape=[emb_dim, emb_dim], initializer=initializer, regularizer=regularizer)
        attn_p = tf.get_variable(name="attention_p", dtype=tf.float32,
            shape=[emb_dim], initializer=initializer, regularizer=regularizer)
        attn_b = tf.get_variable(name="attention_b", dtype=tf.float32,
            shape=[emb_dim], initializer=initializer, regularizer=regularizer)

        for i in range(0, attr_size):
            for j in range(i+1, attr_size):
                element_wise_prod_list.append(
                    tf.multiply(uattr_emb[:, i, :], uattr_emb[:, j, :]))
                count += 1

        element_wise_prod = tf.stack(element_wise_prod_list, axis=1,
            name="afm_element_wise_prof")  # b * (k*(k-1)) * d
        interactions = tf.reduce_sum(element_wise_prod, axis=2, 
            name="afm_interactions")  # b * (k*(k-1))
        num_interactions = attr_size * (attr_size - 1) / 2  # aka: k *(k-1)

        # attentional part
        attn_mul = tf.reshape(
            tf.matmul(tf.reshape(
                element_wise_prod, shape=[-1, emb_dim]), attn_W),
            shape=[-1, num_interactions, emb_dim])  # b * (k*k-1)) * d

        attn_relu = tf.reduce_sum(
            tf.multiply(attn_p, tf.nn.relu(attn_mul + attn_b)), axis=2, keepdims=True)
        # after relu/multiply: b*(k*(k-1))*d; 
        # after reduce_sum + keepdims: b*1*d

        attn_out = tf.nn.softmax(attn_relu)

        afm = tf.reduce_sum(tf.multiply(attn_out, element_wise_prod), axis=1, name="afm")
        # afm: b*(k*(k-1))*d => b*d
        if dropout_keep:
            afm = tf.nn.dropout_keep(afm, dropout_keep)

        return afm, attn_out 


def centroid(hidden_enc, n_centroid, emb_size, tao, name_scope, var_name, corr_metric,
             regularizer=None, activation=None):
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

def mlp(raw_data, layers, name_scope, regularizer=None):
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
s

def get_embeddings(vocab_size, num_units, name_scope, zero_pad=False):
    """Construct a embedding matrix

    Args:
        vocab_size - vocabulary size (the V.)
        num_units - the embedding size (the d.)
        name_scope - the name scope of the matrix
        zero_pad - [bool] whether to pad the matrix by column of zeros

    Returns:
        embedding matrix - [float] (V+1, d)
    """

    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable('embedding_matrix', dtype=tf.float32,
            shape=[vocab_size, num_units],
            initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                embeddings[1:, :]), 0)

    return embeddings
