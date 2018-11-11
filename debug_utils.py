# debug_utils.py

import numpy as np
import tensorflow as tf

def show_when_wrong(data, label, pred, step):
    num_wrong = 0
    length = label.shape[0]

    for i in range(length):
        if np.argmax(label[i]) != np.argmin(pred[i]):
            print (i, data[i], label[i], pred[i])
            num_wrong += 1

    print('After ', step, 'runs, total:', length, 'wrong', num_wrong)
    return num_wrong


def show_when_not_close(data, label, pred, step):
    num_wrong = 0
    length = label.shape[0]
    margin = tf.constant(0.2, dtype=tf.float64)

    for i in range(length):
        abs_diff = tf.reduce_max(tf.abs(tf.subtract(data[i], pred[i])))
        diff = abs_diff.eval()
        if diff > 0.4:
            print(i,'abs_diff:', diff, data[i], pred[i])
            num_wrong += 1

    print('After ', step, 'runs, total:', length, 'wrong', num_wrong)
    return num_wrong
