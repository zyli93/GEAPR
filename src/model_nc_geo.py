"""Model for IRSFN

    @author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

    Notes:
        get_embeddings:
            Emb mat cardinality add one (+1) because the user/item/attribute
            id are from 1 to # of user. Adding one to avoid overflow.
"""

import sys

import tensorflow as tf
from modules import get_embeddings
from modules import autoencoder, gatnet, attentional_fm
from pandas import read_csv
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize


class IRSModel:
    """docstring"""
    def __init__(self, flags):
        """Build a model with basic Auto Encoder

        Args:
            flags - FLAGS from the main function
        """
        self.F = flags

        # Placeholders
        self.is_train = tf.compat.v1.placeholder(shape=(), dtype=tf.bool, name="training_flag")
        self.batch_user = tf.compat.v1.placeholder(shape=[None, ], dtype=tf.int32, name="batch_user")
        self.batch_pos = tf.compat.v1.placeholder(shape=[None, ], dtype=tf.int32, name="batch_pos_item")
        self.batch_neg = tf.compat.v1.placeholder(shape=[None, ], dtype=tf.int32, name="batch_neg_item")  # (b*nsr)
        self.batch_uf = tf.compat.v1.placeholder(shape=[None, self.F.num_total_user+1],
            dtype=tf.int32, name="batch_user_friendship")
        self.batch_usc = tf.compat.v1.placeholder(shape=[None, self.F.num_total_user+1],
            dtype=tf.float32, name="batch_user_struc_ctx")
        self.batch_uattr = tf.compat.v1.placeholder(shape=[None, self.F.afm_num_field],
            dtype=tf.int32, name="batch_user_attribute")
        print(self.batch_user)
        print(self.batch_pos)
        print(self.batch_neg)
        print(self.batch_uf)
        print(self.batch_usc)
        print(self.batch_uattr)

        # fetch-ables
        self.loss, self.losses = None, None # overall loss
        self.user_centroids, self.item_centroids = None, None  # ctrds
        self.train_op = None
        self.test_scores = None
        self.uf_attns, self.uattr_attns1, self.uattr_attns2 = None, None, None
        self.pos_item_ct_logits, self.neg_item_ct_logits = None, None
        self.user_emb_agg_attn = None
        self.user_ct_logits = None
        self.train_op = None
        self.optim_ops, self.optim_dict = None, None

        self.output_dict = {}

        # global step counter
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.inc_gs_op = tf.compat.v1.assign(self.global_step, self.global_step+1)

        # optimizer
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.F.learning_rate)

        self.poi_inf_mat = self.load_poi_inf_mat()
        self.poi_inf_mat = tf.convert_to_tensor(self.poi_inf_mat, tf.float32, name="poi_influence")
        self.num_girds = self.F.num_lat_grid * self.F.num_long_grid

        # build
        ub_data, (ub_row, ub_col), ub_shape = self.load_user_poi_adj_mat()
        indices_list = list(zip(list(ub_row), list(ub_col)))
        self.sp_ub_adj_mat = tf.compat.v1.SparseTensor(
            indices=indices_list, values=list(ub_data), dense_shape=ub_shape)
        self.sp_ub_adj_mat = tf.sparse.to_dense(self.sp_ub_adj_mat, default_value=0)

        # build graph
        self.build_graph()

    def load_poi_inf_mat(self):
        """load poi influence matrix"""
        print("\t[model] loading poi influence matrix")
        in_file = "./data/parse/yelp/citycluster/{}/business_influence_scores.csv".format(self.F.yelp_city)
        df = read_csv(in_file, header=None)
        assert df.shape[0] == self.F.num_total_item + 1
        assert df.shape[1] == self.F.num_long_grid * self.F.num_lat_grid
        return df.values

    def load_user_poi_adj_mat(self):
        """load user-poi adjacency matrix"""
        print("\t[model] loading user-poi adjacency matrix")
        in_file = "./data/parse/yelp/citycluster/{}/city_user_business_adj_mat.npz".format(self.F.yelp_city)
        sp_mat = load_npz(in_file)  # coo
        sp_mat = normalize(sp_mat, norm="l1", axis=1)  # after normalize coo -> csr
        sp_mat = sp_mat.tocoo()
        assert sp_mat.shape[0] == self.F.num_total_user + 1
        assert sp_mat.shape[1] == self.F.num_total_item + 1
        return sp_mat.data, (sp_mat.row, sp_mat.col), sp_mat.shape

    def build_graph(self):
        """
        Three basic options
            1 - AE: auto encoders
            2 - GAT: graph attention network
            3 - AFM: attentional factorization machine
        """

        # regularizer and initializer 
        reglr = tf.contrib.layers.l2_regularizer(scale=self.F.regularization_weight)
        inilz = tf.contrib.layers.xavier_initializer()

        # ===========================
        #      Item embedding
        # ===========================
        # item embedding is moved to the first place
        item_emb_mat = get_embeddings(vocab_size=self.F.num_total_item+1,
                                      num_units=self.F.hid_rep_dim, var_scope="item",
                                      zero_pad=True)  #
        pos_item_emb = tf.nn.embedding_lookup(item_emb_mat, self.batch_pos)  # (b,h)
        neg_item_emb = tf.nn.embedding_lookup(item_emb_mat, self.batch_neg)  # (b*nsr,h)

        # ===========================
        #      Auto Encoders
        # ===========================
        usc_rep, ae_recons_loss = autoencoder(input_features=self.batch_usc,
            layers=self.F.ae_layers, var_scope="ae",
            regularizer=reglr, initializer=inilz)

        # ===========================
        #   Graph Attention Network
        # ===========================
        # m: # of poi's; sp_ub_adj_mat: (n+1, m+1)
        user_emb_mat = tf.matmul(self.sp_ub_adj_mat, item_emb_mat)  # (n, d)

        uf_rep, self.uf_attns = gatnet(
            var_scope="gat", embedding_mat=user_emb_mat, is_training=self.is_train,
            adj_mat=self.batch_uf, input_indices=self.batch_user, hid_rep_dim=self.F.hid_rep_dim,
            n_heads=self.F.gat_nheads,
            ft_drop=self.F.gat_ft_dropout, attn_drop=self.F.gat_coef_dropout)

        # ===========================
        #      Attention FM
        # ===========================
        uattr_rep, self.uattr_attns1, self.uattr_attns2 = attentional_fm(
            var_scope="afm", input_features=self.batch_uattr, is_training=self.is_train,
            emb_dim=self.F.embedding_dim, feat_size=self.F.afm_num_total_user_attr+1,
            initializer=inilz, regularizer=reglr,
            use_dropout=self.F.afm_use_dropout, dropout_rate=self.F.afm_dropout_rate,
            hid_rep_dim=self.F.hid_rep_dim, attr_size=self.F.afm_num_field)

        # ==========================
        #    Geo preference
        # ==========================
        user_geo_pref_emb_mat = get_embeddings(vocab_size=self.F.num_total_user+1,
            num_units=self.num_girds, var_scope="geo", zero_pad=True)  # (n+1,n_grid)

        ugeo_rep = tf.nn.embedding_lookup(user_geo_pref_emb_mat, self.batch_user)  # (b, n_grid)
        # ugeo_rep = tf.nn.relu(ugeo_rep)  # commented for not helping
        ip_geo_rep = tf.nn.embedding_lookup(self.poi_inf_mat, self.batch_pos)
        in_geo_rep = tf.nn.embedding_lookup(self.poi_inf_mat, self.batch_neg)

        # ==========================
        #      Merging factors
        # ==========================
        with tf.compat.v1.variable_scope("attn_agg"):
            uf_rep = tf.layers.dense(uf_rep, units=self.F.hid_rep_dim, activation=tf.nn.relu)
            usc_rep = tf.layers.dense(usc_rep, units=self.F.hid_rep_dim, activation=tf.nn.relu)
            uattr_rep = tf.layers.dense(uattr_rep, units=self.F.hid_rep_dim, activation=tf.nn.relu)
            module_fan_in = [uf_rep, usc_rep, uattr_rep]

            # module_fan_in = [uf_rep]
            # module_fan_in = [uattr_rep]
            # module_fan_in = [usc_rep]
            self.output_dict['module_fan_in'] = module_fan_in
            user_emb = tf.stack(values=module_fan_in, axis=1)  # (b,3,h)
            user_emb_attn = tf.layers.dense(user_emb, units=1, activation=tf.nn.relu,
                    use_bias=False, kernel_initializer=inilz)  # (b,3,1)
            user_emb_attn = tf.nn.softmax(user_emb_attn, axis=1)  # (b,3,1)
            self.user_emb_agg_attn = tf.squeeze(user_emb_attn)  # (b,3), VIZ
            user_emb = tf.squeeze(
                tf.matmul(user_emb, user_emb_attn, transpose_a=True))  # (b,h)

        """
        user_emb - (b,d)
        ugeo_rep - (b,ngrid)
        
        pos_item_emb - (b,d)
        neg_item_emb - (b*nsr, d)
        """
        user_emb = tf.concat([user_emb, ugeo_rep], axis=1, name="geo_user_emb")
        neg_item_emb = tf.concat([neg_item_emb, in_geo_rep], axis=1, name="neg_geo_item_emb")
        pos_item_emb = tf.concat([pos_item_emb, ip_geo_rep], axis=1, name="pos_geo_item_emb")



        print(user_emb)
        print(neg_item_emb)
        print(pos_item_emb)

        """Without centroid:
            use: user_emb, pos_item_emb, and neg_item_emb"""

        # ======================
        #       Losses
        # ======================
        if self.F.loss_type == "ranking":
            # inner product + subtract
            pos_interactions = tf.reduce_sum(
                # tf.multiply(user_ct_rep, item_ct_reps[0]), axis=-1)  # (b)
                tf.multiply(user_emb, pos_item_emb), axis=-1)  # (b)
            pos_interactions = tf.tile(pos_interactions,
                                       multiples=[self.F.negative_sample_ratio])  # (b*nsr)

            user_rep_tiled = tf.tile(user_emb,
                multiples=[self.F.negative_sample_ratio, 1])  # (b*(nsr+1),h)
            neg_interactions = tf.reduce_sum(
                tf.multiply(user_rep_tiled, neg_item_emb), axis=-1)  # (b*nsr)

            negpos_diff = tf.subtract(neg_interactions, pos_interactions)

            self.output_dict["negpos_diff"] = negpos_diff

            # final loss V1: using hinge loss
            # final_loss = tf.reduce_sum(tf.reduce_max(negpos_diff, 0), name="ranking_cls_loss")

            # final loss V2: using RankNet loss
            # o_ij = ( - negpos_diff)
            # - (- negpos_diff) + log(1 + exp(-negpos_diff))
            #   => log(1 + exp(negpos_diff))
            rank_loss = tf.reduce_sum(
                tf.math.log(1 + tf.math.exp(negpos_diff)))

        elif self.F.loss_type == "binary":
            # inner product + sigmoid
            p_size = self.F.batch_size
            n_size = self.F.batch_size * self.F.negative_sample_ratio
            ground_truth = tf.constant([1]*p_size+[0]*n_size, dtype=tf.float32,
                shape=[p_size+n_size])  # (p_size+n_size)

            user_ct_rep_tiled = tf.tile(user_emb,
                multiples=[self.F.negative_sample_ratio+1, 1])
            # shape: (b*nsr, d)
            item_ct_reps_concat = tf.concat([pos_item_emb,  neg_item_emb], axis=0)
            logits = tf.reduce_sum(
                tf.multiply(user_ct_rep_tiled, item_ct_reps_concat), axis=-1)
            rank_loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=ground_truth, logits=logits, name="binary_cls_loss"))

        else:
            raise ValueError("Specify loss type to be `ranking` or `binary`")

        # loss = rank_loss +
        #        [now disabled] ae_reconstruction +
        #        centroid_correlation +
        #        regularization

        # ae_loss = self.F.ae_recon_loss_weight * ae_recons_loss
        # ct_loss = self.F.ctrd_corr_weight * (user_ct_corr_cost + item_ct_corr_cost)
        rg_loss = tf.compat.v1.losses.get_regularization_loss()

        # self.losses = [rank_loss, ct_loss, rg_loss]
        self.losses = [rank_loss, rg_loss]
        self.loss = tf.reduce_sum(self.losses)

        # ======================
        #   Optimization Ops
        # ======================

        if not self.F.separate_loss:
            self.optim_ops = self.optimizer.minimize(
                self.loss, global_step=self.global_step)
            self.optim_ops = [self.optim_ops]

        # all_var_scopes = ["ae", "gat", "afm", ["attn_agg", "centroids"]]

        # set a few alias
        else:
            get_coln = tf.compat.v1.get_collection
            adm_optim = tf.compat.v1.train.AdamOptimizer
            TRN_VAR = tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES

            all_vars = [get_coln(TRN_VAR, scope=x) for x in ["ae", "gat", "afm"]]
            all_vars.append(get_coln(TRN_VAR, "attn_agg") + get_coln(TRN_VAR, "centroids"))

            self.optim_ops = [
                adm_optim(self.F.learning_rate).minimize(self.loss, var_list=x)
                for x in all_vars]

        # ======================
        #   Generate score
        # ======================
        """Generate score for testing time
        input: user_ct_rep, item_emb_mat"""

        # all_item_ct_rep, _ = centroid(
        #     input_features=item_emb_mat, n_centroid=self.F.num_item_ctrd,
        #     emb_size=self.F.hid_rep_dim, tao=self.F.tao, var_scope="centroids",
        #     var_name="item_centroids", activation=self.F.ctrd_activation,
        #     regularizer=reglr)  # (n,h)

        all_item_rep = tf.concat([item_emb_mat, self.poi_inf_mat], axis=1)

        self.test_scores = tf.matmul(user_emb, all_item_rep, transpose_b=True)  # (b,n)
