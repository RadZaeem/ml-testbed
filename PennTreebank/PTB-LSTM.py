#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Raden Muaz
# adapted from tensorpack PTB example

# TODO binarize properly all 8 weights.
import dorefa
from dorefa import get_dorefa
import rnn_dorefa
import numpy as np
import os
import argparse

# from tensorflow.contrib.keras.backend import * #clip function

from tensorpack import *
from tensorpack.tfutils import optimizer, summary, gradproc
from tensorpack.utils import logger
from tensorpack.utils.fs import download, get_dataset_path
from tensorpack.utils.argtools import memoized_ignoreargs


from tensorpack import *
from tensorpack.tfutils.symbolic_functions import prediction_incorrect
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu

import reader as tfreader
from reader import ptb_producer

import tensorflow as tf
rnn = tf.contrib.rnn

SEQ_LEN = 35
HIDDEN_SIZE = 650
NUM_LAYER = 2
BATCH = 20
DROPOUT = 0.5
VOCAB_SIZE = None
TRAIN_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt'
VALID_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt'
TEST_URL = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt'


BITW = 1
BITA = 2
BITG = 6
TOTAL_BATCH_SIZE = 128
BATCH_SIZE = None

@memoized_ignoreargs
def get_PennTreeBank(data_dir=None):
    if data_dir is None:
        data_dir = get_dataset_path('ptb_data')
    if not os.path.isfile(os.path.join(data_dir, 'ptb.train.txt')):
        download(TRAIN_URL, data_dir)
        download(VALID_URL, data_dir)
        download(TEST_URL, data_dir)
    word_to_id = tfreader._build_vocab(os.path.join(data_dir, 'ptb.train.txt'))
    data3 = [np.asarray(tfreader._file_to_word_ids(os.path.join(data_dir, fname), word_to_id))
             for fname in ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']]
    return data3, word_to_id


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.int32, (None, SEQ_LEN), 'input'),
                InputDesc(tf.int32, (None, SEQ_LEN), 'nextinput')]

    def _build_graph(self, inputs):
        tf.set_random_seed(1)
        is_training = get_current_tower_context().is_training
        input, nextinput = inputs
        initializer = tf.random_uniform_initializer(-0.05, 0.05)
        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        # monkey-patch tf.get_variable to apply fw
        def new_get_variable(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv0' in name or 'fct' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)
        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)    # still use relu for 32bit cases
            # logger.info("Binarizing activation {}".format(x))
            return tf.clip_by_value(x, 0.0, 1.0)
        def activate(x):
            return fa(nonlin(x))


        def get_basic_cell():
            cell = rnn_dorefa.BasicLSTMCell(fg=fg, num_units=HIDDEN_SIZE, activation=activate, forget_bias=0.0, reuse=tf.get_variable_scope().reuse)
            if is_training:
                cell = rnn.DropoutWrapper(cell, output_keep_prob=DROPOUT)
            return cell

        with remap_variables(new_get_variable):#, \
                #argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                #argscope([rnn.BasicLSTMCell, rnn.MultiRNNCell], use_bias=False, nl=tf.identity):
            cell = rnn_dorefa.MultiRNNCell([get_basic_cell() for _ in range(NUM_LAYER)])
            #print("cell is:" + repr(cell))

            def get_v(n):
                return tf.get_variable(n, [BATCH, HIDDEN_SIZE],
                                       trainable=False,
                                       initializer=tf.constant_initializer())
            self.state = state_var = \
                (rnn.LSTMStateTuple(get_v('c0'), get_v('h0')),
                 rnn.LSTMStateTuple(get_v('c1'), get_v('h1')))
            # cell = rnn_dorefa.MultiRNNCell([get_basic_cell() for _ in range(NUM_LAYER)])



            embeddingW = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE], initializer=initializer)
            input_feature = tf.nn.embedding_lookup(embeddingW, input)  # B x seqlen x hiddensize
            input_feature = Dropout(input_feature, DROPOUT)

            with tf.variable_scope('LSTM', initializer=initializer):
                input_list = tf.unstack(input_feature, num=SEQ_LEN, axis=1)  # seqlen x (Bxhidden)
                # outputs = []
                outputs, last_state = rnn_dorefa.static_rnn(cell, input_list, state_var, scope='rnn',fg=fg)#.apply(fb,state_var)
                # for o in outs: outputs.append(fg(o))

            # update the hidden state after a rnn loop completes
            update_state_ops = [
                tf.assign(state_var[0].c, last_state[0].c),
                tf.assign(state_var[0].h, last_state[0].h),
                tf.assign(state_var[1].c, last_state[1].c),
                tf.assign(state_var[1].h, last_state[1].h)]

            # last_state=fg(last_state)

            # seqlen x (Bxrnnsize)
            output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])  # (Bxseqlen) x hidden
            logits = FullyConnected('fc', output, VOCAB_SIZE, nl=tf.identity, W_init=initializer, b_init=initializer)
            xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=tf.reshape(nextinput, [-1]))

            with tf.control_dependencies(update_state_ops):
                self.cost = tf.truediv(tf.reduce_sum(xent_loss),
                                       tf.cast(BATCH, tf.float32), name='cost')  # log-perplexity

            perpl = tf.exp(self.cost / SEQ_LEN, name='perplexity')
            summary.add_moving_summary(perpl, self.cost)

    def reset_lstm_state(self):
        s = self.state
        z = tf.zeros_like(s[0].c)
        return tf.group(s[0].c.assign(z),
                        s[0].h.assign(z),
                        s[1].c.assign(z),
                        s[1].h.assign(z), name='reset_lstm_state')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1.0, trainable=False)
        opt = tf.train.GradientDescentOptimizer(lr)
        return optimizer.apply_grad_processors(
            opt, [gradproc.GlobalNormClip(5)])


def get_config():
    logger.auto_set_dir()

    data3, wd2id = get_PennTreeBank()
    global VOCAB_SIZE
    VOCAB_SIZE = len(wd2id)
    steps_per_epoch = (data3[0].shape[0] // BATCH - 1) // SEQ_LEN

    train_data = TensorInput(
        lambda: ptb_producer(data3[0], BATCH, SEQ_LEN),
        steps_per_epoch)
    val_data = TensorInput(
        lambda: ptb_producer(data3[1], BATCH, SEQ_LEN),
        (data3[1].shape[0] // BATCH - 1) // SEQ_LEN)

    test_data = TensorInput(
        lambda: ptb_producer(data3[2], BATCH, SEQ_LEN),
        (data3[2].shape[0] // BATCH - 1) // SEQ_LEN)

    M = Model()
    return TrainConfig(
        data=train_data,
        model=M,
        callbacks=[
            ModelSaver(),
            HyperParamSetterWithFunc(
                'learning_rate',
                lambda e, x: x * 0.80 if e > 6 else x),
            RunOp(lambda: M.reset_lstm_state()),
            InferenceRunner(val_data, [ScalarStats(['cost'])]),
            RunOp(lambda: M.reset_lstm_state()),
            InferenceRunner(
                test_data,
                [ScalarStats(['cost'], prefix='test')], tower_name='InferenceTowerTest'),
            RunOp(lambda: M.reset_lstm_state()),
            CallbackFactory(
                trigger=lambda self:
                [self.trainer.monitors.put_scalar(
                    'validation_perplexity',
                    np.exp(self.trainer.monitors.get_latest('validation_cost') / SEQ_LEN)),
                 self.trainer.monitors.put_scalar(
                     'test_perplexity',
                     np.exp(self.trainer.monitors.get_latest('test_cost') / SEQ_LEN))]
            ),
        ],
        max_epoch=70,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    launch_train_with_config(config, SimpleTrainer())
