import tensorflow as tf
from time import localtime, strftime

time_str = strftime("%j-%H-%M", localtime())
LOG_DIR = '/tmp/tboard/simpleEx/'
LOG_PREFIX = time_str


def my_saver():

    #Prepare to feed input, i.e. feed_dict and placeholders
    """
    w1 = tf.placeholder("float", name="w1")
    w2 = tf.placeholder("float", name="w2")
    b1= tf.Variable(2.0,name="bias")
    feed_dict ={w1:4,w2:8}
    """

    w1= tf.Variable(10.0, name="w1")
    w2= tf.Variable(2.0, name="w2")
    b1= tf.Variable(2.0,name="bias")


    #Define a test operation that we will restore
    w3 = tf.add(w1,w2)
    w4 = tf.multiply(w3,b1,name="op_to_restore")

    saver = tf.train.Saver([w1, w2, b1])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #Run the operation by feeding input
    print (sess.run(w4,feed_dict=None))
    #Prints 24 which is sum of (w1+w2)*b1

    #Now, save the graph
    #saver.save(sess, LOG_DIR + LOG_PREFIX, global_step=1)
    saver.save(sess, LOG_DIR, global_step=1)
    return


def my_restorer():
    sess = tf.Session()
    # First let's load meta graph and restore weights

    #saver = tf.train.import_meta_graph(LOG_DIR + LOG_PREFIX + '-1.meta')
    saver = tf.train.import_meta_graph(LOG_DIR + '-1.meta')
    saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR + '/'))

    # Access saved Variables directly
    print(sess.run('bias:0'))
    # This will print 2, which is the value of bias that we saved


    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    graph = tf.get_default_graph()
    print('graph:', graph)
    w1 = graph.get_tensor_by_name("w1:0")
    w2 = graph.get_tensor_by_name("w2:0")
    print(w1, w2)
    feed_dict = {w1: 13.0, w2: 17.0}

    # Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

    x = sess.run(op_to_restore, feed_dict)
    print('x = ', x)
    # This will print 60 which is calculated
    return


def restore_and_print_vars():
    sess = tf.Session()
    #new_saver = tf.train.import_meta_graph(LOG_DIR + LOG_PREFIX + '-1.meta')
    new_saver = tf.train.import_meta_graph(LOG_DIR + '-1.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR + '/'))
    all_vars = tf.get_collection('vars')
    print('Saved Vars:')
    for v in all_vars:
        v_ = sess.run(v)
        print('Var:', v_)

if __name__ == '__main__':
    my_saver()
    my_restorer()
    restore_and_print_vars()


