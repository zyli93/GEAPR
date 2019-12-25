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
    print("\n=======================")
    print("\t\tID:{}".format(F.trial_id))
    print("=======================")

    # training
    with tf.Session(config=config) as sess, \
         open(perfdir + "/" + F.trial_id + ".perf", "w") as perf_writer:

        # initialization
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # batch data generator
        trn_iter = dataloader.data_batch_generator(set_="train")

        # run epochs
        for epoch in range(F.epoch):
            data_loader.has_next = True

            while trn_iter.has_next:

                # get batch
                bU, bI, bUn, bIn = data_loader.generate_batch()

                # TODO: create indices, values, shapes

                # run training operation
                loss, pred = sess.run(
                    fetches=[
                        model.loss,
                        model.predictions
                    ],
                    feed_dict={
                        model.adjU_batch_sp: (biU, bvU, b_shapeU),
                        model.adjI_batch_sp: (biI, bvI, b_shapeI),
                        model.labels: batch_label
                    }
                )

                msg = None

                # print results and write to file
                if sess.run(model.global_step) \
                        % F.log_n_iter == 0:

                    # TODO: get acc
                    # TODO: get precision, recall
                    acc = None
                    precision, recall = None, None

                    msg = build_msg(stage="Trn",
                                    epoch=epoch,
                                    iteration=data_loader.batch_index,
                                    global_step=sess.run(model.global_step),
                                    acc=acc,
                                    prec=precision,
                                    rec=recall)

                    # write to file
                    print(msg, file=perf_writer)

                    # print performance every 1000 batches
                    if sess.run(model.global_step) % (10 * F.log_n_iter) == 0:
                        print(msg)

                # save model
                if sess.run(model.global_step) \
                        % F.save_n_epoch == 0:
                    print("\tSaving Checkpoint at global step [{}]!"
                          .format(sess.run(model.global_step)))
                    saver.save(sess,
                               save_path=logdir,
                               global_step=sess.run(model.global_step))

            # run validation set
            epoch_msg = evaluate(sess=sess,
                                 data_loader=data_loader,
                                 epoch=epoch,
                                 model=model)

            print(epoch_msg)
            print(epoch_msg, file=perf_writer)

    print("Training finished!")


def validation(model, sess, epoch, data_loader):
    """run validation"""


def evaluate(model, data_loader):
    """ Evaluation function

    Args:
        model - the model
        sess - the session used to run everything
        epoch - number of epochs
        data_loader - the data loader

    Return:
        msg - a message made report the message
    """

    return "empty msg"

