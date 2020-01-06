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
        afm - attentional factorization machine output
        attn_out - attention output 

    """

    with tf.variable_scope(name_scope) as scope:
        embedding_mat = get_embeddings(vocab_size=feat_size, num_units=emb_dim,
            name_scope=scope, zero_pad=True)  # (|A|+1, d) lookup table for all attr emb 
        uattr_emb = tf.nn.embedding_lookup(embedding_mat, input_features)  # (b, k, d)
        element_wise_prod_list = []

        attn_W = tf.get_variable(name="attention_W", dtype=tf.float32,
            shape=[emb_dim, emb_dim], initializer=initializer, regularizer=regularizer)
        attn_p = tf.get_variable(name="attention_p", dtype=tf.float32,
            shape=[emb_dim], initializer=initializer, regularizer=regularizer)
        attn_b = tf.get_variable(name="attention_b", dtype=tf.float32,
            shape=[emb_dim], initializer=initializer, regularizer=regularizer)

        for i in range(0, attr_size):
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
        # after reduce_sum + keepdims: b*(k*(k-1))*1

        attn_out = tf.nn.softmax(attn_relu)  # b*(k*(k-1)*d

        afm = tf.reduce_sum(tf.multiply(attn_out, element_wise_prod), axis=1, name="afm")
        # afm: b*(k*(k-1))*d => b*d
        if dropout_keep:
            afm = tf.nn.dropout_keep(afm, dropout_keep)

        attn_out = tf.squeeze(attn_out, name="attention_output")

        # TODO: first order feature not considered yet!

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


def gatnet(name_scope, embedding_mat, input_indices, num_nodes, in_rep_size,
        adj_mat):
    """Graph Attention Network component for users/items

    Code adapted from: https://github.com/PetarV-/GAT
    But only implemented a simple (one-layered) version

    Notations:
        b - batch size
        n - total number of nodes (user-friendship graph)

    Args:
        name_scope - name scope
        embedding_mat - the whole embedding matrix of nodes
        input_features - (?, ?, ?) the inputs of batch user indices
        num_nodes - [int] total number of nodes in the graph
        adj_mat - [int] (b, n) adjacency matrix for the batch
        intn_rep_size - internal representation size

    Notes:
        1. How to get bias_mat from adj_mat (learned from GAT repo issues)?
            - adj_mat, bool or int of (0, 1)
            - adj_mat, cast to float32
            - 1 - adj_mat, 0 => 1 and 1 => 0
            - -1e9 * (above), 0 => -1e9 and 1 => 0
            - obtained bias_mat
    """

    with tf.name_scop(name_scope) as scope:

        input_features = tf.nn.embedding_lookup(embedding_mat, input_indices)
        bias_mat = -1e9 * (1 - tf.cast(adj_mat, dtype=tf.float32))  # (b, n)

        # TODO: what is ffd_drop, in_drop, attn_drop, residual

        attns = []

        for _ in range(n_heads[0]):
            attns.append(gat_attn_head(seq=input_features, bias_mat=bias_mat,
                output_size=in_rep_size, activation=tf.nn.relu, in_drop=ffd_drop,
                coef_drop=attn_drop, residual=False))
            h_1 = tf.concat(attns, axis=-1)

        out = []

        # TODO: is the following useful?
        for i in range(n_heads[-1]):
            out.append(gat_attn_head(seq=h_1, bias_mat, output_size=???,
                activation=lambda x: x, in_drop=ffd_drop, coef_drop=attn_drop,
                residual=False))

        logits = tf.add_n(out) / n_heads[-1]

        return logits  # TODO: what else to return?


def gat_attn_head(seq, output_size, bias_mat, activation,
    in_drop=0.0, coef_drop=0.0, residual=False):
    """Single graph attention head

    Notations:
        b - batch size
        n - total node size
        k - feature size (embedding/representation size)
        oz - output size

    Args:
        seq - (b, n, k) input data in format of batch adj-mat
        output_size - (oz) output size (internal representation size)
        bias_mat - (b, n, n) bias (or mask) matrix (0 for edges, 1e-9 for non-edges)
        activation - activation function
        in_drop - what's this?
        coef_drop - what's this?
        residual - residual from the previous layers

    Returns:
        ???
    """

    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # this is a mat mult
        seq_fts = tf.layers.conv1d(seq, output_size, 1, use_bias=False)  # (b, n, oz)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)  # (b, n. 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)  # (b, n, 1)
        logtis=  f_1 + tf.transpose(f_2, [0, 2, 1])  # (b, n, n)
        coefs = tf.nn.softmax(tf.nn,leaky_relu(logits) + bias_mat)  # (b, n, n)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)

        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
            # why drop twice?

            vals = tf.matmul(coefs, seq_fts)  # TODO: why use matmul?
        ret = tf.contrib.layers.bias_add(vals)

        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + tf.conv1d(seq, ret.shape[-1], 1)
            else:
                ret = ret + seq

        return activation(ret)


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


# ======== not used ==========

def mlp(raw_data, layers, name_scope, regularizer=None):
    """Multi-layer Perceptron

    :param raw_data:
    :param layers: [raw_dim, layer1, layer2, ...]
    :param name_scope:
    :param regularizer:
    :return:
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
