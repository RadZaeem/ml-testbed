#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

G = tf.get_default_graph()

# def p_ternarize(x, p):

#     x = tf.tanh(x)
#     shape = x.get_shape()

#     thre = tf.get_variable('T', trainable=False, collections=[tf.GraphKeys.VARIABLES, 'thresholds'],
#             initializer=0.05)
#     flat_x = tf.reshape(x, [-1])
#     k = int(flat_x.get_shape().dims[0].value * (1 - p))
#     topK, _ = tf.nn.top_k(tf.abs(flat_x), k)
#     update_thre = thre.assign(topK[-1])
#     tf.add_to_collection('update_thre_op', update_thre)

#     mask = tf.zeros(shape)
#     mask = tf.select((x > thre) | (x < -thre), tf.ones(shape), mask)

#     with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
#         w =  tf.sign(x) * tf.stop_gradient(mask)

#     tf.histogram_summary(w.name, w)
#     return w

def tw_ternarize(x, thre):

    print("before ternarized name:", repr(x))
    shape = x.get_shape()
    print("\nShape of")
    print(shape)

    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre)

    w_p = tf.get_variable('Wp',  collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'positives'], initializer=1.0)
    w_n = tf.get_variable('Wn',  collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'negatives'], initializer=1.0)


    tf.summary.scalar(w_p.name, w_p)
    tf.summary.scalar(w_n.name, w_n)

    mask = tf.ones(shape)
    mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
    mask_np = tf.where(x < -thre_x, tf.ones(shape) * w_n, mask_p)
    mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

    with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w =  tf.sign(x) * tf.stop_gradient(mask_z)

    w = w * mask_np


    print("Ternarized name: ", repr(w))
    tf.summary.histogram(w.name, w)

    # tf.summary(w.name, w)

    return w



def tw_ternarize_bias(x, thre):

    print("bias name:",repr(x))
    shape = x.get_shape()
    print("\nShape of")
    print(shape)

    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre)

    w_p = tf.get_variable('Bp',  collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'positives'], initializer=1.0)
    w_n = tf.get_variable('Bn',  collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'negatives'], initializer=1.0)


    tf.summary.scalar(w_p.name, w_p)
    tf.summary.scalar(w_n.name, w_n)

    mask = tf.ones(shape)
    mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
    mask_np = tf.where(x < -thre_x, tf.ones(shape) * w_n, mask_p)
    mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

    with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w =  tf.sign(x) * tf.stop_gradient(mask_z)

    w = w * mask_np


    print("Ternarized name: ", repr(w))
    tf.summary.histogram(w.name, w)

    # tf.summary(w.name, w)

    return w

def tw_ternarize_input(x, thre):
    print("input name:", repr(x))
    shape = x.get_shape()
    print("\nShape of")
    print(shape)

    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre)

    w_p = tf.get_variable('Ip',  collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'positives'], initializer=1.0)
    w_n = tf.get_variable('In',  collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'negatives'], initializer=1.0)


    tf.summary.scalar(w_p.name, w_p)
    tf.summary.scalar(w_n.name, w_n)

    mask = tf.ones(shape)
    mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
    mask_np = tf.where(x < -thre_x, tf.ones(shape) * w_n, mask_p)
    mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

    with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w =  tf.sign(x) * tf.stop_gradient(mask_z)

    w = w * mask_np


    print("Ternarized name: ", repr(w))
    tf.summary.histogram(x.op.name, w)

    # tf.summary(w.name, w)

    return w
