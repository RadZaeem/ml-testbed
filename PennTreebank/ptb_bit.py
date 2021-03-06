#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Raden Muaz
# adapted from tensorpack PTB example
import dorefa
# from dorefa import get_dorefa
import bit_rnn
import bit_utils
import numpy as np
import os
import argparse


from tensorpack import *
from tensorpack.tfutils import optimizer, summary, gradproc
from tensorpack.utils import logger
from tensorpack.utils.fs import download, get_dataset_path
from tensorpack.utils.argtools import memoized_ignoreargs

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

# class Config(object):
#     learning_rate = 1e-3
#     max_grad_norm = 5#10
#     num_layers = NUM_LAYER#2
#     num_steps = SEQ_LEN #20
#     hidden_size = HIDDEN_SIZE#300
#     max_epoch = 100
#     keep_prob = DROPOUT#0.5
#     batch_size = BATCH#20
#     vocab_size = 10000
#     nr_epoch_first_stage = 40
#     nr_epoch_second_stage = 80
#     w_bit = 2
#     f_bit = 2
#     cell_type = 'lstm'
class Config(object):
    learning_rate = 1e-3
    max_grad_norm = 10
    num_layers = 1
    num_steps = 20
    hidden_size = 300
    max_epoch = 100
    keep_prob = 0.5
    batch_size = 20
    vocab_size = 10000
    nr_epoch_first_stage = 40
    nr_epoch_second_stage = 80
    w_bit = 2
    f_bit = 2
    cell_type = 'lstm'

class Model(ModelDesc):
    def _get_inputs(self):
        conf = Config()
        return [InputDesc(tf.int32, (None, conf.num_steps ), 'input'),
                InputDesc(tf.int32, (None, conf.num_steps), 'nextinput')]

    def _build_graph(self, inputs):
        conf = Config()

        is_training = get_current_tower_context().is_training
        input, nextinput = inputs
        initializer = tf.uniform_unit_scaling_initializer()
        # initializer = tf.random_uniform_initializer(-0.05, 0.05)

        def get_basic_cell():
            # cell = rnn.BasicLSTMCell(num_units=conf.hidden_size, forget_bias=0.0, reuse=tf.get_variable_scope().reuse)
            cell = bit_rnn.BitLSTMCell(num_units=conf.hidden_size, w_bit=conf.w_bit, f_bit=conf.f_bit,#)
                forget_bias=0.0, reuse=tf.get_variable_scope().reuse)
            if is_training and conf.keep_prob < 1:
                cell = bit_rnn.DropoutWrapper(cell, output_keep_prob=conf.keep_prob)
            return cell

        cell = rnn.MultiRNNCell([get_basic_cell() for _ in range(conf.num_layers)])


        def get_v(n):
            return tf.get_variable(n, [conf.batch_size, conf.hidden_size],#,[BATCH, HIDDEN_SIZE],
                                   trainable=False,
                                   initializer=tf.constant_initializer())

        self.state = state_var = \
            (rnn.LSTMStateTuple(get_v('c0'), get_v('h0')),
             rnn.LSTMStateTuple(get_v('c1'), get_v('h1')))
        embeddingW = tf.get_variable('embedding', [conf.vocab_size, conf.hidden_size], 
            initializer=initializer)#tf.random_uniform_initializer)
        input_feature = tf.nn.embedding_lookup(embeddingW, input)  # B x seqlen x hiddensize

        print("\n-> Input Rounding")
        input_feature = bit_utils.round_bit(tf.nn.relu(input_feature), bit=conf.f_bit)
        
        if is_training and conf.keep_prob < 1:
            input_feature = Dropout(input_feature, conf.keep_prob)

        # print("\n\nThe STATE:")
        # print(self.state)


        with tf.variable_scope('LSTM', initializer=initializer):
            input_list = tf.unstack(input_feature, num=conf.num_steps, axis=1)  # seqlen x (Bxhidden)
            outputs, last_state = rnn.static_rnn(cell, input_list, state_var,
             scope='rnn')


        update_state_ops = []
        for k in range(conf.num_layers):
            update_state_ops.extend([
                tf.assign(state_var[k].c, last_state[k].c),
                tf.assign(state_var[k].h, last_state[k].h)])

        def replace_w(x):
            # if x.op.name.endswith('Matrix'):
            if x.op.name.endswith('W'):
                print ("\nKERNEL Before quantize name: " + x.op.name)
                return bit_utils.quantize_w(tf.tanh(x), bit=conf.w_bit)
            elif x.op.name.endswith('b'):
                print ("\nbias Before round name: " + x.op.name)
                # tf.summary.histogram(x.name, x)
                return x
                return bit_utils.round_bit_whist(x, bit=conf.w_bit)
            else:
                print ("\nNOT Quantizing:" + x.op.name)
                tf.summary.histogram(x.name, x)
                return x

        # seqlen x (Bxrnnsize)
        output = tf.reshape(tf.concat(outputs, 1), [-1, conf.hidden_size])  # (Bxseqlen) x hidden
        with bit_utils.replace_variable(replace_w):
                # lambda x: bit_utils.quantize_w(tf.tanh(x), bit=conf.w_bit)):
            logits = FullyConnected('fc', output, conf.vocab_size, nl=tf.identity, W_init=initializer, b_init=initializer)
        # logits = FullyConnected('fc', output, conf.vocab_size, nl=tf.identity, W_init=initializer, b_init=initializer)
        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.reshape(nextinput, [-1]))

        with tf.control_dependencies(update_state_ops):
            self.cost = tf.truediv(tf.reduce_sum(xent_loss),
                                   tf.cast(conf.batch_size, tf.float32), name='cost')  # log-perplexity

        perpl = tf.exp(self.cost / conf.num_steps, name='perplexity')
        summary.add_moving_summary(perpl, self.cost)

    def reset_lstm_state(self):
        conf = Config()
        s = self.state
        z = tf.zeros_like(s[0].c)
        print("\n==> Zeroing state\n")
        z = bit_utils.round_bit(tf.sigmoid(z), bit=conf.f_bit)
        # print("\nResetting state\n")
        return tf.group(s[0].c.assign(z),
                        s[0].h.assign(z),
                        s[1].c.assign(z),
                        s[1].h.assign(z), name='reset_lstm_state')

    def _get_optimizer(self):
        conf = Config()
        lr = tf.get_variable('learning_rate', initializer=conf.learning_rate, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        tf.summary.scalar('learning_rate', lr)
        return optimizer.apply_grad_processors(
            opt, [gradproc.GlobalNormClip(conf.max_grad_norm)])



def get_config():
    logger.auto_set_dir()

    data3, wd2id = get_PennTreeBank()
    # global VOCAB_SIZE

    conf = Config()

    # VOCAB_SIZE = len(wd2id)
    steps_per_epoch = (data3[0].shape[0] // conf.batch_size - 1) // conf.num_steps

    train_data = TensorInput(
        lambda: ptb_producer(data3[0], conf.batch_size, conf.num_steps),
        steps_per_epoch)
    val_data = TensorInput(
        lambda: ptb_producer(data3[1], conf.batch_size, conf.num_steps),
        (data3[1].shape[0] // conf.batch_size - 1) // conf.num_steps)

    test_data = TensorInput(
        lambda: ptb_producer(data3[2], conf.batch_size, conf.num_steps),
        (data3[2].shape[0] // conf.batch_size - 1) // conf.num_steps)


    def get_learning_rate(epoch, base_lr):
        print("\n\nEpoch, "+repr(epoch) +", " + repr(base_lr))
        print(base_lr)
        if epoch <= 40:# conf.nr_epoch_first_stage:
            return base_lr
        elif epoch <= 80:#conf.nr_epoch_second_stage:
            return base_lr * 0.1
        else:
            return base_lr * 0.01

    M = Model()
    from tensorflow.python import debug as tf_debug
    return TrainConfig(
        data=train_data,
        model=M,
        callbacks=[
            # HookToCallback(tf_debug.LocalCLIDebugHook()),
            ModelSaver(),
            HyperParamSetterWithFunc(
                'learning_rate', #lambda e, x:  1e-3),
                get_learning_rate),
                # lambda e, x: x * 0.80 if e > 6 else x),
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
                    np.exp(self.trainer.monitors.get_latest('validation_cost') / conf.num_steps)),
                 self.trainer.monitors.put_scalar(
                     'test_perplexity',
                     np.exp(self.trainer.monitors.get_latest('test_cost') / conf.num_steps))]
            ),
        ],
        max_epoch=conf.max_epoch,
    )


if __name__ == '__main__':
    tf.set_random_seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    

    # launch_train_with_config(config, SimpleTrainer())
    SyncMultiGPUTrainer(config).train()

