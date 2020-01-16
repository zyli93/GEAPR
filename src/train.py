"""Trainer file

    @author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import numpy as np
import tensorflow as tf
from utils import build_msg
from rank_metrics import metrics_poi


def train(flags, model, dataloader):
    """ Trainer function
    Args:
        flags - container of all settings
        model - the model we use
        dataloader - the data loader providing all input data
    """

    F = flags
    ckpt_dir = "./ckpt/{}_{}/".format(F.trial_id, F.dataset)
    perf_file = "./performance/{}.perf".format(F.trial_id)

    # === Saver ===
    saver = tf.train.Saver(max_to_keep=10)

    # === Configurations ===
    config = tf.ConfigProto(
        allow_soft_placement=True
        # , log_device_placement=True
    )
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # === Run training ===
    print("=======================")
    print("\t\tExperiment ID:{}".format(F.trial_id))
    print("=======================")

    # training
    with tf.Session(config=config) as sess, \
            open(perf_file, "w") as perf_writer:

        # initialization
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # batch data generator
        trn_iter = dataloader.get_train_batch_iterator()

        # run epochs
        for epoch in range(F.epoch):
            # abbrevs: batch index, batch user, batch positive and negative items
            # bI: scalar; bU: (batch_size,1)
            # bP: (batch_size, 1); bN: (batch_size * nsr, 1)
            for bI, bU, bP, bN in trn_iter:
                bUf, bUsc = dataloader.get_user_graphs(bU)
                bUattr = dataloader.get_user_attributes(bU)
                print("print shape of bUf, bUsc, bUattr")
                print(bUf.shape, bUsc.shape, bUattr)

                # run training operation
                _, gs, loss = sess.run(
                    fetches=[model.train_op, model.global_step, model.loss],
                    feed_dict={
                        model.is_training: True, model.batch_user: bU,
                        model.batch_pos: bP, model.batch_neg: bN,
                        model.batch_uf: bUf, model.batch_usc: bUsc,
                        model.batch_uattr: bUattr})

                # print results and write to file
                if gs and not(gs % F.log_per_iter):
                    msg_loss = build_msg(stage="Trn", ep=epoch, gs=gs, bi=bI, loss=loss)
                    print(msg_loss, file=perf_writer)  # write to log
                    if gs % (10 * F.log_per_iter) == 0:
                        print(msg_loss)

                # save model, only when save model flag is on
                if F.save_model and gs and not(gs % F.save_per_iter):
                    print("\tSaving Checkpoint at global step [{}]!"
                          .format(sess.run(model.global_step)))
                    saver.save(sess, save_path=ckpt_dir, global_step=gs)

            # run validation set
            eval_dict = evaluate(False, model, dataloader, F, sess)
            msg_val_score = build_msg("Val", epoch=epoch, eval_dict=eval_dict)

            print(msg_val_score)
            print(msg_val_score, file=perf_writer)

            # TODO: for debug purpose, have NOT implemented TEST. Pls Implement it!

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

    Return:
        msg - a message made report the message
    """
    bs = F.batch_size
    tv_U, tv_gt = dataloader.get_test_valid_dataset(is_test=is_test)
    scores_list = []
    for i in range(len(tv_U) // bs):
        # tv_: test or validation
        tv_bU = tv_U[i*bs: min((i+1)*bs, len(tv_U))]
        tv_buf, tv_busc = dataloader.get_user_graphs(tv_bU)
        tv_buattr = dataloader.get_user_attributes(tv_bU)

        scores = sess.run(fetches=[model.test_scores],
            feed_dict={
                model.is_training: False,
                model.batch_user: tv_bU, model.batch_uattr: tv_buattr,
                model.batch_uf: tv_buf, model.batch_usc: tv_busc})
        scores_list.append(scores)

    scores = np.concatenate(scores_list, axis=0)
    assert len(scores) == len(tv_gt), \
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
            model.is_training: False,
            model.batch_user: valU, model.batch_uattr: val_uattr,
            model.batch_uf: val_uf, model.batch_usc: val_usc})

    eval_dict = metrics_poi(gt=val_gt, pred_scores=scores, k_list=F.candicate_k)
    msg = build_msg("Trn", epoch=epoch, **eval_dict[F.candicate_k[0]])
    print(msg)
