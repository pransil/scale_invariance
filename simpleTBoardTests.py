# nnet2.py
import numpy as np
import tensorflow as tf
from time import localtime, strftime



if __name__ == '__main__':
    a = tf.constant(5)
    b = tf.constant(10)
    #c_var = tf.get_variable(name='c_name', dtype=tf.int8, shape=[1])
    c_var = a + b

    time_str = strftime("%j-%H-%M", localtime())
    LOG_DIR = '/tmp/tboard/simple/' + time_str

    saver = tf.train.Saver({'c_name': c _var})
    #tf.summary.scalar('accuracy', c)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        graph_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        summary, c_var =sess.run([merged,c_var])

        graph_writer.add_summary(summary, 1)
        saver.save(sess, LOG_DIR, global_step=1)

        print ('c_var =',c_var.eval())