"""Model for IRSFN

    @author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import tensorflow as tf
from modules import get_embeddings
from modules import autoencoder, gatnet, attentional_fm
from modules import centroid, centroid_corr

# TODO: user location 
# TODO: unify emb_dim and internal representation


class IRSModel:
    def __init__(self, flags):
        """Build a model with basic Auto Encoder

        Args:
            flags - FLAGS from the main function
        """
        self.F = flags

        # Placeholders
        self.batch_user = tf.placeholder(
            shape=[None], dtype=tf.int32, name="batch_user")
        self.batch_pos = tf.placeholder(
            shape=[None], dtype=tf.int32, name="batch_pos_item")
        self.batch_neg = tf.placeholder(
            shape=[None], dtype=tf.int32, name="batch_neg_item")  # (b*nsr)
        self.batch_uf = tf.placeholder(shape=[None, self.F.num_total_user], 
            dtype=tf.int32, name="batch_user_friendship")
        self.batch_usc = tf.placeholder(shape=[None, self.F.num_total_item], 
            dtype=tf.float32, name="batch_user_struc_ctx")
        self.batch_uattr = tf.placeholder(shape=[None, self.F.num_user_attr], 
            dtype=tf.int32, name="batch_user_attribute")

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

        # regularizer and initializer 
        reglr = tf.contrib.layers.l2_regularizer(scale=F.regularization_weight)
        inilz = tf.contrib.layers.xavier_initializer()


        # ===========================
        #      Auto Encoders
        # ===========================

        usc_rep = autoencoder(input_features=self.batch_usc,
            layers=self.F.ae_layers name_scope="user_struc_context_ae",
            regularizer=reglr, initializer=inilz)

        # ===========================
        #   Graph Attention Network
        # ===========================

        user_emb_mat = get_embeddings(vocab_size=self.F.num_total_user,
            num_units=self.F.embedding_size, name_scope="gat", zero_pad=True)

        uf_rep = gatnet(name_scope="gat", embedding_mat=user_emb_mat,
            adj_mat=self.batch_uf, input_indices=self.batch_user,
            num_nodes=self.F.num_total_user, in_rep_size=self.F.rep_dim,
            n_heads=self.F.gat_nheads, ft_drop=self.F.gat_ft_dropout,
            attn_drop=self.F.gat_coef_dropout)


        # ===========================
        #      Attention FM
        # ===========================
        
        uattr_rep, afm_attn_output = attentional_fm(
            name_scope="afm", input_features=self.batch_uattr,
            emb_dim=self.F.embedding_dim, feat_size=self.F.XXX,
            initializer=inilz, regularizer=reglr, dropout_keep=self.F.XXX)

        # ===========================
        #      Item embedding
        # ===========================

        item_emb_mat = get_embeddings(vocab_size=self.F.num_total_item,
            num_units=self.F.embedding_dim, name_scope="item_embedding_matrix", 
            zero_pad=True)
        pos_item_emb = tf.nn.embedding_lookup(item_emb_mat, self.batch_pos)  # (b,d)
        neg_item_emb = tf.nn.embedding_lookup(item_emb_mat, self.batch_neg)  # (b*nsr,d)

        # ==========================
        #      User embedding
        # ==========================

        user_emb = uf_rep, usc_rep, uattr_rep # TODO: combine this

        # ============================
        #   Centroids/Interests/Cost
        # ============================

        u_ct_rep = centroid(input_features=user_emb, 
            n_centroid=self.F.num_user_ctrd, emb_size=self.F.embedding_size,
            tao=self.F.tao, name_scope="centroids", var_name="user_centroids",
            activation=self.F.ctrd_activation, regularizer=reger) # (b,d)

        i_ct_reps = []  # 0,pos; 1,neg
        for x_item_emb in [pos_item_emb, neg_item_emb]:
            tmp_ct_rep = centroid(input_features=tmp_item_emb,
                n_centroid=self.F.num_item_ctrd, emb_size=self.F.embedding_size,
                tao=self.F.tao, name_scope="centroids", var_name="item_centroids",
                activation=self.F.ctrd_activation, regularizer=reger)
            i_ct_reps.append(tmp_ct_rep)

        with tf.variable_scope("centroids", reuse=tf.AUTO_REUSE):
            self.user_centroids = tf.get_variable(name="user_centroids")
            self.item_centroids = tf.get_variable(name="item_centroids")

        u_ct_corr_cost = centroid_corr(self.user_centroids)
        i_ct_corr_cost = centroid_corr(self.item_centroids)

        # ======================
        #       Losses
        # ======================

        # TODO: loss part, whether to use activation?

        if self.F.loss_type == "ranking":
            # inner product + subtract
            pos_interactions = tf.reduce_sum(
                tf.multiply(u_ct_rep, i_ct_reps[0]), axis=-1)
            pos_interactions = tf.tile(pos_interactions,
                multiples=[self.F.negative_sample_ratio])  # (b*neg)

            neg_interactions = tf.reduce_sum(
                tf.multiply(u_ct_rep, i_ct_reps[1]), axis=-1)  # (b*neg)
            final_loss = tf.reduce_sum(tf.subtract(neg_interactions, pos_interactions),
                name="ranking_cls_loss")

        elif self.F.loss_type == "binary":
            # inner product + sigmoid
            p_size = self.F.batch_size
            n_size = self.F.batch_size * self.F.negative_sample_ratio
            ground_truth = tf.constant([1]*p_size + [0]*n_size, dtype=tf.int32,
                shape=[p_size+n_size])  # (p_size+n_size)

            u_ct_rep_tiled = tf.tile(u_ct_rep,
                multiples=[self.F.negative_sample_ratio, 1])
            # shape: (b*neg_sample, d)
            i_ct_reps_concat = tf.concat(i_ct_reps, axis=0)
            logits = tf.reduce_sum(
                tf.multiply(u_ct_rep_tiled, i_ct_reps_concat), axis=-1)
            final_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=ground_truth, logits=logits, name="binary_cls_loss")

        else:
            raise ValueError("Specify loss type to be `ranking` or `binary`")

        # loss = final_loss + 
        #        ae_reconstruction + centroid_correlation + regularization
        loss = final_loss
        loss += self.F.ae_recon_loss_weight * (ae_lossU + ae_lossI)
        loss += self.F.ctrd_corr_weight * (u_ct_corr_cost + i_ct_corr_cost)
        loss += tf.losses.get_regularization_losses()

        # TODO: correct way to update loss

        self.loss = loss
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=self.global_step)


