"""Trainer file

    @author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from utils import build_msg, make_dir
from rank_metrics import metrics_poi


def train(flags, model, dataloader):
    """ Trainer function
    Args:
        flags - container of all settings
        model - the model we use
        dataloader - the data loader providing all input data
    """

    F = flags
    ckpt_dir = "./output/ckpt/{}_{}/".format(F.trial_id, F.dataset)
    perf_file = "./output/performance/{}_{}.perf".format(F.trial_id, F.dataset)

    # === Saver ===
    saver = tf.compat.v1.train.Saver(max_to_keep=10)

    # === Configurations ===
    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True
        # , log_device_placement=True
    )
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # tf.reset_default_graph()
    tf.set_random_seed(F.random_seed)
    np.random.seed(F.random_seed)

    # === Run training ===
    print("===" * 18)
    print("\t\tExperiment ID:{}".format(F.trial_id))
    print("===" * 18)

    # training
    with tf.compat.v1.Session(config=config) as sess, \
            open(perf_file, "w") as perf_writer:

        # initialization
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        # run epochs
        for epoch in range(F.epoch):
            # batch data generator
            trn_iter = dataloader.get_train_batch_iterator()

            # abbrevs: batch index, batch user, batch positive and negative items
            # bI: scalar; bU: (batch_size,1)
            # bP: (batch_size, 1); bN: (batch_size * nsr, 1)
            for bI, bU, bP, bN in trn_iter:
                # break  # used to debug for evaluation
                bUf, bUsc = dataloader.get_user_graphs(bU)
                bUsc, bUf = bUsc.toarray(), bUf.toarray()
                bUattr = dataloader.get_user_attributes(bU)

                feed_dict = {
                    model.is_train: True, model.batch_user: bU,
                    model.batch_pos: bP, model.batch_neg: bN,
                    model.batch_uf: bUf, model.batch_usc: bUsc,
                    model.batch_uattr: bUattr}

                # run training operation, update global step
                _, _, test = sess.run(
                    fetches=[model.inc_gs_op] + model.optim_ops + [model.test],
                    feed_dict=feed_dict)

                # print(test)

                # print results and write to file
                if bI and not(bI % F.log_per_iter):
                    # compute loss and output
                    gs, odict, loss, losses = sess.run(
                        fetches=[model.global_step,
                                 model.output_dict, model.loss, model.losses],
                        feed_dict=feed_dict)
                    msg_loss = build_msg(stage="Trn", ep=epoch, gs=gs, bi=bI, loss=loss)

                    print(msg_loss)

                    print("losses")
                    print(losses)

                    # for i, x in enumerate(odict["module_fan_in"]):
                    #     print(i, x.max(), x.min(), x.mean(), x.var())

                    print(msg_loss, file=perf_writer)  # write to log
                    # if bI % (10 * F.log_per_iter) == 0:
                    #     print(msg_loss)

                # save model, only when save model flag is on
                if F.save_model and bI and not(bI % F.save_per_iter):
                    print("\tSaving Checkpoint at global step [{}]!"
                          .format(sess.run(model.global_step)))
                    saver.save(sess, save_path=ckpt_dir, global_step=gs)

            # run test set
            if epoch == 6:
                eval_dict = evaluate(True, model, dataloader, F, sess)
                msg_test_score = build_msg("Tst", epoch=epoch, eval_dict=eval_dict)
                print(msg_test_score)
                print(msg_test_score, file=perf_writer)

                user_geo_pref = sess.run(fetches=model.ugeo_pref_mat)
                np.save("./attentions/user_pref", user_geo_pref)

    print("Training finished!")


def evaluate(is_test, model, dataloader, F, sess):
    """ Testing/validation function

    Args:
        is_test - [flag] of `test` (True)  or `validation` (False)
        model - the model
        sess - the session used to run everything
        epoch - the number of epochs of this evaluation
        dataloader - the data loader
        F - the flags
        mat_gt - matrix version of the ground truth

    Return:
        msg - a message made report the message

    Notes:
        1 - Jan21: changed load batch from `for` to tqdm(`for`)
    """
    bs = F.batch_size
    tv_U, tv_gt = dataloader.get_test_valid_dataset(is_test=is_test)

    scores_list = []

    gat_attentions = []
    afm_order1_attentions = []
    afm_order2_attentions = []
    feat_importance = []

    print("\Running Evaluation")
    # for i in tqdm(range(len(tv_U) // bs + 1)):
    for i in range(len(tv_U) // bs + 1):
        if i * bs >= len(tv_U):
            break
        # tv_: test or validation
        tv_bU = tv_U[i*bs: min((i+1)*bs, len(tv_U))]
        tv_buf, tv_busc = dataloader.get_user_graphs(tv_bU)
        tv_buf, tv_busc = tv_buf.toarray(), tv_busc.toarray()
        tv_buattr = dataloader.get_user_attributes(tv_bU)

        if F.task == "perf":
            b_scores = sess.run(
                fetches=model.test_scores,
                feed_dict={
                    model.is_train: False,
                    model.batch_user: tv_bU, model.batch_uattr: tv_buattr,
                    model.batch_uf: tv_buf, model.batch_usc: tv_busc})
            scores_list.append(b_scores)
        else:
            b_scores, b_out_dict = sess.run(
                fetches=[model.test_scores, model.output_dict],
                feed_dict={
                    model.is_train: False,
                    model.batch_user: tv_bU, model.batch_uattr: tv_buattr,
                    model.batch_uf: tv_buf, model.batch_usc: tv_busc})
            scores_list.append(b_scores)

            print("gat_attn")
            print(len(b_out_dict['gat_attn']))
            print(b_out_dict["gat_attn"][0].shape)
            gat_attentions.append(b_out_dict['gat_attn'][0])

            print("afm attn order1")
            print(b_out_dict["afm_attn_order1"].shape)
            afm_order1_attentions.append(b_out_dict['afm_attn_order1'])
            print("afm attn order2")
            print(b_out_dict["afm_attn_order2"].shape)
            afm_order2_attentions.append(b_out_dict['afm_attn_order2'])

            print("feat_importance_attn")
            print(b_out_dict["feat_importance_attn"].shape)
            feat_importance.append(b_out_dict['feat_importance_attn'])

    # TODO: add geolocation
    scores = np.concatenate(scores_list, axis=0)

    if F.task == "inter":
        gat_attentions = np.concatenate(gat_attentions, axis=0)
        afm_order1_attentions = np.concatenate(afm_order1_attentions, axis=0)
        afm_order2_attentions = np.concatenate(afm_order2_attentions, axis=0)
        feat_importance = np.concatenate(feat_importance, axis=0)

        dir_ = "./attentions/"
        make_dir(dir_)

        np.save(dir_+"gat", gat_attentions)
        np.save(dir_+"afm1", afm_order1_attentions)
        np.save(dir_+"afm2", afm_order2_attentions)
        np.save(dir_+"feat", feat_importance)
        np.save(dir_+"user", tv_U)
        np.save(dir_+"gt", tv_gt)

    assert scores.shape[0] == len(tv_gt), \
        "[evaluate] sizes of scores and ground truth don't match"
    eval_dict = metrics_poi(gt=tv_gt, pred_scores=scores, k_list=F.candidate_k)

    return eval_dict


def validation(model, epoch, sess, dataloader, F):
    """run validation on sampled test sets
    
    [Not used] merge to evaluate
    """
    valU, val_gt = dataloader.get_test_valid_dataset(is_test=False)
    val_uf, val_usc = dataloader.get_user_graphs(valU)
    val_uattr = dataloader.get_user_attributes(valU)

    # score (b, n+1)
    scores = sess.run(fetches=[model.test_scores],
        feed_dict={
            model.is_train: False,
            model.batch_user: valU, model.batch_uattr: val_uattr,
            model.batch_uf: val_uf, model.batch_usc: val_usc})

    eval_dict = metrics_poi(gt=val_gt, pred_scores=scores, k_list=F.candicate_k)
    msg = build_msg("Trn", epoch=epoch, **eval_dict[F.candicate_k[0]])
    print(msg)
