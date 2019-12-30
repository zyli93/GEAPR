"""Model for irsfn

    @author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import tensorflow as tf
from modules import autoencoder, centroid


class IRSModel:
    def __init__(self, flags):
        """Build a model with basic Auto Encoder

        Args:
            flags - FLAGS from the main function
        """
        self.F = flags

        # Placeholders
        self.batch_user = tf.placeholder(
            [None, 1], dtype=tf.int32, name="batch_user")
        self.batch_pos = tf.placeholder(
            [None, 1], dtype=tf.int32, name="batch_pos_item")
        self.batch_neg = tf.placeholder(
            [None, flags.negative_sample_ratio], dtype=tf.int32, name="batch_neg_item")
        self.batch_uf = tf.placeholder() # TODO: fix me!
        self.batch_usc = tf.placeholder(
            [None, flags.num_total_item], dtype=tf.float32, name="batch_user_struc_context")
        self.batch_uattr = tf.placeholder(
            [None, flags.num_user_attr], dtype=tf.int32, name="batch_user_attribute")

        # fetch-ables
        self.loss = None  # overall loss
        self.predictions = None  # predictions of a batch

        # 
        self.user_centroids, self.item_centroids = None, None  # ctrds
        self.train_op = None

        # global step counter
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.F.learning_rate)

        # build graph
        self.build_graph()


    def build_graph(self):
        """
        Three basic options
            1 - AE: auto encoders
            2 - GAT: graph attention network
            3 - AFM: attentional factorization machine
        """

        #
        reglr = tf.contrib.layers.l2_regularizer(scale=F.regularization_weight)


        # ===========================
        #      Auto Encoders
        # ===========================



        # ===========================
        #   Graph Attention Network
        # ===========================


        # ===========================
        #       Attention FM
        # ===========================


        # TODO: embU, embI directly send to centroids?

        # ============================
        #      Centroids/Interest
        # ============================

        ctrdU, corr_costU = centroid(hidden_enc=embU,
                                     n_centroid=F.user_n_ctrd,
                                     emb_size=F.embedding_size,
                                     tao=F.tao,
                                     name_scope="user_attn_pool",
                                     var_name="user_centroids",
                                     corr_metric=F.corr_metric,
                                     activation=F.ctrd_act_func,
                                     regularizer=reger)

        ctrdI, corr_costI = centroid(hidden_enc=embI,
                                     n_centroid=F.item_n_ctrd,
                                     emb_size=F.embedding_size,
                                     tao=F.tao,
                                     name_scope="item_attn_pool",
                                     var_name="item_centroids",
                                     corr_metric=F.corr_metric,
                                     activation=F.ctrd_act_func,
                                     regularizer=reger)

        # TODO:
        # Notes: autoencoder does not return reg loss,
        #        get sum of reg loss by tf.losses.get_regularization_loss()

        # ======================
        #       Prediction
        # ======================

        # ctrdU, ctrdI \in (b,d)


        # ======================
        #       Losses, TODO
        # ======================

        loss = pred_loss1  # prediction loss
        loss += F.ae_weight * (ae_lossU + ae_lossI)  # auto encoder reconstruction loss
        loss += F.corr_weight * (corr_costU + corr_costI)  # centroids/interests correlation loss
        loss += tf.losses.get_regularization_losses()  # get all regularization

        self.loss = loss

        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=self.global_step)
        # ======================
        #      The Centroids
        # ======================

        # var_scope names are in centroids part
        with tf.variable_scope("user_attn_pool", reuse=True):
            self.user_centroids = tf.get_variable(dtype=tf.float32,
                                                  name="user_centroids")

        with tf.variable_scope("item_attn_pool", reuse=True):
            self.item_centroids = tf.get_variable(dtype=tf.float32,
                                                  name="item_centroids")

