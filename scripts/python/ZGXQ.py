import tensorflow as tf
import tensorflow.contrib.layers as tc
import os
import sys
import random
import numpy as np

MAX_ITER = 1000000
BATCH_SIZE = 512
START_LEARNING_RATE = 0.0015
TRAIN_KEEP_PROB = 0.5
IS_TRAINING = True


class DataSet(object):
    def __init__(self, input_path, batch_size):
        self.input_path = input_path
        self.raw_data = None
        self.batch_index = 0
        self.batch_size = batch_size
        self.data_size = 0
        self.prepare_data()

    def prepare_data(self):
        count = 0
        fp = open("./result/3.txt", "r")
        while 1:
            buffer = fp.read(8 * 1024 * 1024)
            if not buffer:
                break
            count += buffer.count('\n')
        fp.close()

        self.raw_data = np.zeros((count, 10 * 9 * 8), 'int32')
        f = open('./result/3.txt')
        nLineIndex = 0
        for line in f:
            arr = line.split(',')
            arrBoard = arr[0].split(';')
            nBoardIndex = 0
            nCurQiPos = int(arrBoard[nBoardIndex])
            for nPos in range(90):
                if nBoardIndex < len(arrBoard) - 1 and nCurQiPos == nPos:
                    nQi = int(arrBoard[nBoardIndex + 1])
                    self.raw_data[nLineIndex][nPos * 7 + abs(nQi) - 1] = nQi / abs(nQi)
                    nBoardIndex += 2
                    if nBoardIndex < len(arrBoard) - 1:
                        nCurQiPos = int(arrBoard[nBoardIndex])
            nMovePos = int(arr[1])
            self.raw_data[nLineIndex][10 * 9 * 7 + nMovePos] = 1
            nLineIndex += 1
        # self.raw_data = np.loadtxt(self.input_path, delimiter=",")
        # self.input = raw_data[:, 0:10 * 9 * 7]
        # self.label = raw_data[:, 10 * 9 * 7:0]
        self.data_size = count
        random.shuffle(self.raw_data)
        print ("File loaded with %d entries " % self.data_size)

    def get_batch(self):
        is_reset = False
        batch_data = self.raw_data[self.batch_index:(self.batch_index + self.batch_size), :]
        batch_input = batch_data[:, 0:10 * 9 * 7]
        batch_label = batch_data[:, 10 * 9 * 7:]
        self.batch_index += self.batch_size
        if (self.batch_index + self.batch_size) > (self.data_size - 1):
            random.shuffle(self.raw_data)
            self.batch_index = 0
            is_reset = True

        return batch_input, batch_label, is_reset


data_set = DataSet("zNup.npz", BATCH_SIZE)

with tf.name_scope("inputs"):
    board_input = tf.placeholder(tf.float32, [None, 10 * 9 * 7], "Board_Input")
    board_label = tf.placeholder(tf.float32, [None, 10 * 9], "Board_Label")

with tf.variable_scope("config"):
    keep_prob = tf.Variable(float(TRAIN_KEEP_PROB), trainable=False, name="keep_prob")
    learning_rate = tf.Variable(float(START_LEARNING_RATE), trainable=False, name="lr")

with tf.name_scope("inputs_reshape"):
    board_input_reshape = tf.reshape(board_input, [-1, 10, 9, 7], name="board_input_reshape")

with tf.name_scope("conv1"):
    conv1 = tc.conv2d(board_input_reshape, 64, kernel_size=[2, 2])
    conv1_pool = tc.max_pool2d(conv1, kernel_size=[2, 2], stride=[1, 1])
    # conv1_pool = tf.nn.dropout(conv1_pool, 0.5)

with tf.name_scope("conv2"):
    conv2 = tc.conv2d(conv1_pool, 64, kernel_size=[2, 2])
    conv2_pool = tc.max_pool2d(conv2, kernel_size=[2, 2], stride=[1, 1])
    # conv2_pool = tf.nn.dropout(conv2_pool, keep_prob=0.5)

with tf.name_scope("fc1"):
    fc1 = tf.nn.dropout(tc.fully_connected(tf.reshape(conv2_pool, [-1, 14 * 4 * 64]), 256), keep_prob=0.5)

with tf.name_scope("fc2"):
    fc2 = tf.nn.dropout(tc.fully_connected(fc1, 256), keep_prob=0.5)

with tf.name_scope("output"):
    board_output = tc.fully_connected(fc1, 90, activation_fn=None)

with tf.name_scope("loss"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(board_output, board_label))
    train_step = tf.train.AdamOptimizer(0.0015).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(board_output, 1), tf.argmax(board_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.name_scope("predict_result"):
    predict_result = tf.argmax(board_output, 1)

with tf.name_scope("assign_lr"):
    lr_decay_op = learning_rate.assign(learning_rate * 0.5)

# Init Session
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state('savedmodel/')
if ckpt and ckpt.model_checkpoint_path:
    # tf.train.import_meta_graph('model.meta')
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('restore model')

writer = tf.summary.FileWriter("logs-ZGXQ", sess.graph)
if IS_TRAINING is False:
    batch_input, _, _ = data_set.get_batch()
    predict_move = sess.run([predict_result], feed_dict={board_input: batch_input})
    print (predict_move)
    quit()
num_epoch = 0
num_run = 0
while num_epoch < MAX_ITER:
    num_run += 1
    batch_input, batch_output, add_epoch = data_set.get_batch()
    if add_epoch is True:
        num_epoch += 1
        saver.save(sess, 'savedmodel/model')
        writer = tf.summary.FileWriter("logs-ZGXQ", sess.graph)
    loss, _ = sess.run([cross_entropy, train_step], feed_dict={board_input: batch_input, board_label: batch_output})
    print ("loss %.4f" % np.average(loss))
    if num_run % 10 == 0:
        accuracy_val = sess.run([accuracy], feed_dict={board_input: batch_input, board_label: batch_output})
        print ("accuracy %f, epoch %d" % (accuracy_val[0], num_epoch))
