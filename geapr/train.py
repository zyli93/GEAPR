"""Trainer file

    @author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""
import sys
sys.path.append(".")

import numpy as np
import tensorflow as tf
from utils import build_msg, make_dir
from rank_metrics import metrics_poi
from tqdm import tqdm


def train(flags, model, dataloader):
    """ Trainer function
    Args:
        flags - container of all settings
        model - the model we use
        dataloader - the data loader providing all input data
    """

    F = flags
    ckpt_dir = "./output/ckpt/{}/".format(F.trial_id)
    perf_file = "./output/performance/{}.perf".format(F.trial_id)

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

                # print results and write to file
                if bI and not(bI % F.log_per_iter):
                    # compute loss and output
                    gs, odict, loss, losses = sess.run(
                        fetches=[model.global_step,
                                 model.output_dict, model.loss, model.losses],
                        feed_dict=feed_dict)
                    msg_loss = build_msg(stage="Trn", ep=epoch, gs=gs, bi=bI, loss=loss)

                    print(msg_loss)
                    print(msg_loss, file=perf_writer)  # write to log

                # save model, only when save model flag is on
                if F.save_model and bI and not(bI % F.save_per_iter):
                    print("\tSaving Checkpoint at global step [{}]!"
                          .format(sess.run(model.global_step)))
                    saver.save(sess, save_path=ckpt_dir, global_step=gs)

            # run test set
            eval_dict = evaluate(model, dataloader, F, sess)
            msg_test_score = build_msg("Tst", epoch=epoch, eval_dict=eval_dict)
            print(msg_test_score)
            print(msg_test_score, file=perf_writer)

    print("Training finished!")


def evaluate(model, dataloader, F, sess):
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
    tv_U, tv_gt = dataloader.get_test_valid_dataset()
    scores_list = []

    print("\Running Evaluation")
    for i in tqdm(range(len(tv_U) // bs + 1)):
        if i * bs >= len(tv_U):
            break
        # tv_: test or validation
        tv_bU = tv_U[i*bs: min((i+1)*bs, len(tv_U))]
        tv_buf, tv_busc = dataloader.get_user_graphs(tv_bU)
        tv_buf, tv_busc = tv_buf.toarray(), tv_busc.toarray()
        tv_buattr = dataloader.get_user_attributes(tv_bU)

        b_scores = sess.run(
            fetches=model.test_scores,
            feed_dict={
                model.is_train: False,
                model.batch_user: tv_bU, model.batch_uattr: tv_buattr,
                model.batch_uf: tv_buf, model.batch_usc: tv_busc})
        scores_list.append(b_scores)

    scores = np.concatenate(scores_list, axis=0)

    assert scores.shape[0] == len(tv_gt), \
        "[evaluate] sizes of scores and ground truth don't match"
    eval_dict = metrics_poi(gt=tv_gt, pred_scores=scores, k_list=F.candidate_k)

    return eval_dict

