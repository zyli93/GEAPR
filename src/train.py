#! /usr/bin/python3
"""
    Trainer file

    Note:
        1. tensorboard is not used in this project

    @author: Zeyu Li <zyli@cs.ucla.edu>

"""

import os
import numpy as np
import tensorflow as tf
from utils import build_msg, cr
from rank_metrics import mapk, ndcg_at_k


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
                print(bUf.shape)
                print(bUsc.shape)
                print(bUattr)

                # run training operation
                _, gs, loss = sess.run(
                    fetches=[model.train_op, model.global_step, model.loss],
                    feed_dict={
                        model.is_training=True, model.batch_user: bU,
                        model.batch_pos: bP, model.batch_neg: bN,
                        model.batch_uf: bUf, model.batch_usc: bUsc,
                        model.batch_uattr: bUattr} )

                # print results and write to file
                if gs % F.log_n_iter == 0:

                    # TODO: get map, ndcg

                    msg = build_msg(stage="Trn", ep=epoch, 
                        gs=gs, bi=bI, map_=map_, ndcg=ndcg)

                    # write to file, print performance every 1000 batches
                    print(msg, file=perf_writer)
                    if gs % (10 * F.log_n_iter) == 0:
                        print(msg)

                # save model
                if gs % F.save_n_poch == 0:
                    print("\tSaving Checkpoint at global step [{}]!"
                          .format(sess.run(model.global_step)))
                    saver.save(sess, save_path=logdir, global_step=gs)

            # run validation set
            epoch_msg = evaluate(sess=sess,
                                 dataloader=dataloader,
                                 epoch=epoch,
                                 model=model)

            print(epoch_msg)
            print(epoch_msg, file=perf_writer)

    print("Training finished!")


def validation(model, sess, dataloader, F):
    """run validation on sampled test sets"""
    valU, val_gt = dataloader.get_test_valid_dataset(is_test=False)
    val_uf, val_usc = dataloader.get_user_graphs(valU)
    val_uattr = dataloader.get_user_attributes(valU)

    # score (b, n+1)
    scores = sess.run(fetches=[model.test_scores],
        feed_dict={
            model.is_training: False,
            model.batch_user: valU, model.batch_uattr: val_uattr,
            model.batch_uf: val_uf, model.batch_usc: bUsc} )

    eval_dict = metrics_poi(ground_truth=val_gt, pred_scores=score, k_list=???)

    # TODO: set k as a list
    


def evaluate(model, dataloader):
    """ Evaluation function

    Args:
        model - the model
        sess - the session used to run everything
        epoch - number of epochs
        dataloader - the data loader

    Return:
        msg - a message made report the message
    """

    # TODO: implement me by batch

    return "empty msg"

