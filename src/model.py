"""
    Model file

    @author: Zeyu Li <zyli@cs.ucla.edu>

    TODO: note that labels should be of integer type

    TODO List:
        1. implement DugrilpSDNE
        2. implement DugrilpGAT
        3. integrate RWR to module build graph
        4. (test) PCA decomposition for GAT
"""

import tensorflow as tf
from modules import autoencoder, centroid, attnnet


class UnInteRec:
    def __init__(self
                 , flags
                 , ae_layers=None):
        """Build a model with basic Auto Encoder

        Args:
            ae_layers - the structure of the Auto Encoder/Decoder
            flags - FLAGS from the main function
        """
        self.F = flags

        # Hyper-params
        self.ae_layers = ae_layers

        # Placeholders
        self.adjU_batch_sp = tf.sparse.placeholder(
            [None, flags.user_size], dtype=tf.int32, name="AdjU_batch_sp")
        self.adjI_batch_sp = tf.sparse.placeholder(
            [None, flags.item_size], dtype=tf.int32, name="AdjU_batch_sp")
        self.labels = tf.placeholder([None, 1], dtype=tf.int32, name="Labels")

        # fetch-ables
        self.loss = None  # overall loss
        self.predictions = None  # predictions of a batch
        self.user_centroids, self.item_centroids = None, None  # ctrds
        self.train_op = None

        # global step counter
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.F.learning_rate)

        # build graph
        self.build_graph()

        """
        # Start with sparse tensors, so commented the dense tensor
        self.adjU_batch = tf.placeholder([None, flags.emb_size],
                                         dtype=tf.int32, name="Adj_U_Batch")
        self.adjI_batch = tf.placeholder([None, flags.emb_size],
                                         dtype=tf.int32, name="Adj_I_Batch")
        """

    def build_graph(self):

        # shorthand of FLAGS
        F = self.F

        #
        reger = tf.contrib.layers.l2_regularizer(scale=F.regularization_weight)

        # ========================
        #       Auto Encoders
        # ========================

        # Notes: autoencoder does not return reg loss,
        #        get sum of reg loss by tf.losses.get_regularization_loss()
        embU, ae_lossU = autoencoder(raw_data=self.adjU_batch_sp,
                                     layers=self.ae_layers[0],
                                     name_scope="user_autoencoder",
                                     regularizer=reger)

        embI, ae_lossI = autoencoder(raw_data=self.adjI_batch_sp,
                                     layers=self.ae_layers[1],
                                     name_scope="item_autoencoder",
                                     regularizer=reger)

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

        # ======================
        #       Prediction
        # ======================

        # ctrdU, ctrdI \in (b,d)

        # predictor 1:
        #   concat two centroids, do binary prediction
        #   TODO: if chosen, add negative samples in training
        with tf.name_scope("prediction_1"):
            concat_ctrd= tf.concat(ctrdI, ctrdU, name="Concat_ui_representation")
            logits1 = tf.layers.dense(inputs=concat_ctrd,
                                      units=F.pred_hid_unit,
                                      activation=tf.nn.relu,
                                      kernel_regularizer=reger,
                                      bias_regularizer=reger)

            # TODO: other layers
            logits1 = tf.layers.dense(inputs=logits1,
                                      units=2,
                                      activation=tf.nn.relu,
                                      kernel_regularizer=reger,
                                      bias_regularizer=reger,
                                      name="pred_logits")  # (b,2)

            pred_loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.one_hot(self.labels, depth=2),
                logits=logits1,
                name="prediction_error_lost"
            )

        self.predictions1 = tf.argmax(logits1, axis=-1, name="predictions")

        # predictor 2:
        #   no negative sampling, compute cross entropy
        #   TODO: if chosen, add negative samples in testing
        with tf.name_scope("predictor"):
            # transform to lower dimension
            ctrdU = tf.layers.dense(inputs=ctrdU,
                                    units=F.pre_hid_unit,
                                    activation=tf.nn.relu,
                                    kernel_regularizer=reger,
                                    bias_regularizer=reger)  # (b, hid_unit)

            # transform to lower dimension
            ctrdI = tf.layers.dense(inputs=ctrdI,
                                    units=F.pre_hid_unit,
                                    activation=tf.nn.relu,
                                    kernel_regularizer=reger,
                                    bias_regularizer=reger)  # (b, hid_unit)

            # normalized
            ctrdU = tf.nn.l2_normalize(ctrdU, axis=1)
            ctrdI = tf.nn.l2_normalize(ctrdI, axis=1)

            # cosine of them, minimize the negative
            pred_loss2 = - tf.reduce_sum(tf.multiply(ctrdU, ctrdI), axis=1)

        self.predictions2 = - pred_loss2

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






