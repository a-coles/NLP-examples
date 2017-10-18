import os
import sys
import collections
import numpy as np

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Get an integer-mapped dictionary
with open('niv_preproc.txt', 'r') as niv:
    words = niv.read().split()
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    training_data = np.array(words)
    training_data = np.reshape(training_data, [-1, ])

# Training Parameters
learning_rate = 0.001
batch_size = 32
display_step = 200 # Show progress after every display_step iterations

# Network Parameters
num_input = 5 # Works on 5-grams
timesteps = 28
num_hidden = 512 # Number of features in the hidden layer
vocab_size = len(dictionary)

# tf Graph input
x = tf.placeholder("float", [None, num_input, 1])
y = tf.placeholder("float", [None, vocab_size])

# Define weight matrix
weights = {'out': tf.Variable(tf.random_normal([num_hidden, vocab_size]))}
# Define bias
biases = {'out': tf.Variable(tf.random_normal([vocab_size]))}

def RNN(x, weights, biases):
   # reshape to [1, num_input]
    x = tf.reshape(x, [-1, num_input])

    # Generate an input of the right length, and consisting of the mapped integers
    x = tf.split(x,num_input,1)

    # Define an LSTM cell
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Define the output of the LSTM cell
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # output * W + b
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Loss - softmax and cross-entropy; optimizer - RMSprop
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Start training
import random
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)
training_steps = 1000 # Could increase

with tf.Session() as sess:
    sess.run(init)
    step = 0
    offset = random.randint(0,num_input+1)
    end_offset = num_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(sess.graph)

    while step < training_steps:
        # Generate a minibatch, somewhat randomly
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, num_input+1)

        # Get the integer representations of the words
        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+num_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, num_input, 1])

        # Generate a one-hot version
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+num_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        # Feed this, optimizer, loss function into the network
        _, acc, loss, onehot_pred = sess.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})

        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + num_input)]
            symbols_out = training_data[offset + num_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            # Show [sequence, of, words, in, batch] - [next] vs. [predicted next]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
        offset += (num_input+1)
    print("Done with optimization.")

    while True:
        prompt = "%s words: " % num_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != num_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(64):
                keys = np.reshape(np.array(symbols_in_keys), [-1, num_input, 1])
                onehot_pred = sess.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")
