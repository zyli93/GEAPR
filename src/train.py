#! /usr/bin/python3
"""
    Trainer file

    Note:
        1. tensorboard is not used in this project

    @author: Zeyu Li <zyli@cs.ucla.edu>
"""

import os
import tensorflow as tf
from utils import build_msg, cr


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
            for bI, bU, bP, bN in trn_iter:
                # TODO: create indices, values, shapes
                bUf, bUsc, bUfnbr = dataloader.get_user_graphs(bU)
                bUattr = dataloader.get_user_attributes(bU)

                # run training operation
                gs, loss = sess.run(
                    fetches=[model.global_step, model.loss],
                    feed_dict={
                        model.batch_user: bU,
                        model.batch_pos: bP, model.batch_neg: bN,
                        model.batch_uf: bUf, model.batch_usc: bUsc,
                        model.batch_uattr: bUattr
                    }
                )

                # print results and write to file
                if sess.run(model.global_step) \
                        % F.log_n_iter == 0:

                    # TODO: get map, ndcg


                    msg = build_msg(stage="Trn", ep=epoch, 
                        gs=gs, bi=bI, map_=map_, ndcg=ndcg)

                    # write to file
                    print(msg, file=perf_writer)

                    # print performance every 1000 batches
                    if sess.run(model.global_step) % (10 * F.log_n_iter) == 0:
                        print(msg)

                # save model
                if sess.run(model.global_step) \
                        % F.save_n_poch == 0:
                    print("\tSaving Checkpoint at global step [{}]!"
                          .format(sess.run(model.global_step)))
                    saver.save(sess,
                               save_path=logdir,
                               global_step=sess.run(model.global_step))

            # run validation set
            epoch_msg = evaluate(sess=sess,
                                 dataloader=dataloader,
                                 epoch=epoch,
                                 model=model)

            print(epoch_msg)
            print(epoch_msg, file=perf_writer)

    print("Training finished!")


def validation(model, sess, epoch, dataloader):
    """run validation"""


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

    return "empty msg"

