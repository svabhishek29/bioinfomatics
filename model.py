import operator
import os
import pickle as pkl
import time
from copy import copy
from sys import stdout

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from util import build_dictionary


class ProteinClassification(object):

    def __init__(self, mode, path, folds, embedding_size, hidden_size, batch_size, keep_prob_dropout,
                 learning_rate, num_epochs=10):

        self.mode = mode
        self.path = path
        self.folds = folds
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.keep_prob_dropout = keep_prob_dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def save_model(self, session, epoch):
        if not os.path.exists('./model/'):
            os.mkdir('./model/')
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        if not os.path.exists('./model/epoch_%d.checkpoint' % epoch):
            saver.save(session, './model/epoch_%d.checkpoint' % epoch)
        else:
            saver.save(session, './model/epoch_%d.checkpoint' % epoch)

    def encoder(self, sentences_embedded, sentences_lengths, dropout):
        with tf.variable_scope("encoder") as varscope:
            cell = tf.contrib.rnn.GRUCell(self.hidden_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout)
            cell = tf.contrib.rnn.MultiRNNCell([cell], state_is_tuple=True)
            sentences_outputs, sentences_states = tf.nn.dynamic_rnn(cell,
                                                                    inputs=sentences_embedded,
                                                                    sequence_length=sentences_lengths,
                                                                    dtype=tf.float32)
            sentences_states_h = sentences_states[-1]

        return sentences_states_h

    def batch_norm_wrapper(self, inputs, is_training, decay=0.999):

        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean, batch_var, beta, scale, 0.0001)
        else:
            return tf.nn.batch_normalization(inputs,
                                             pop_mean, pop_var, beta, scale, 0.0001)

    def run(self):
        self.is_train = (self.mode == 'Train')

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        # Load the training data
        with open('train_data.pkl', 'rb') as f:
            data_sequences = pkl.load(f)
        with open('train_labels.pkl', 'rb') as f:
            data_labels = pkl.load(f)

        dictionary, reverse_dictionary, data_lengths, self.max_seq_len, enc_sequences = build_dictionary(data_sequences)
        print(len(data_lengths), np.max(data_lengths), np.min(data_lengths), np.mean(data_lengths), np.std(data_lengths))
        self.dictionary = sorted(dictionary.items(), key=operator.itemgetter(1))
        print(self.dictionary)
        self.vocabulary_size = len(dictionary)
        self.val_size = len(data_sequences) // self.folds
        fold = 1
        print('Training fold number %d. Each fold of size %d' % (fold, len(data_sequences) // self.folds))

        if self.is_train:
            # self.max_seq_len = 2000
            original_lengths = copy(data_lengths)

            data_sequences = enc_sequences[:, :self.max_seq_len]
            for i in range(len(data_lengths)):
                if data_lengths[i] > self.max_seq_len:
                    data_sequences[i] = np.concatenate(
                        (enc_sequences[i, :self.max_seq_len - 100], enc_sequences[i, -100:]), axis=0)
                    data_lengths[i] = self.max_seq_len

            if self.folds == 1:
                val_mask = np.array([False])
            else:
                val_mask = np.arange(self.val_size * (fold - 1), self.val_size * (fold))

            # Use seed to ensure same randomisation is applied for each fold
            np.random.seed(4)
            perm = np.random.permutation(len(data_sequences))
            data_labels = np.array(data_labels)

            data_sequences = data_sequences[perm]
            data_labels = data_labels[perm]
            data_lenghts = data_lengths[perm]
            original_lengths = original_lengths[perm]

            self.val_data = data_sequences[val_mask]
            self.val_labels = data_labels[val_mask]
            self.val_lengths = data_lengths[val_mask]
            self.val_original_lengths = original_lengths[val_mask]

            self.train_data = np.delete(data_sequences, val_mask, axis=0)
            self.train_labels = np.delete(data_labels, val_mask, axis=0)
            self.train_lengths = np.delete(data_lengths, val_mask, axis=0)
            self.train_original_lengths = np.delete(original_lengths, val_mask, axis=0)

        print('Build model')

        # Define placeholders and variables
        initializer = tf.random_normal_initializer()
        self.word_embeddings = tf.get_variable('embeddings', [self.vocabulary_size, self.embedding_size], tf.float32,
                                               initializer=initializer)
        sequences = tf.placeholder(tf.int32, [None, None], "sequences")
        sequences_lengths = tf.placeholder(tf.int32, [None], "sequences_lengths")
        labels = tf.placeholder(tf.int64, [None], "labels")
        keep_prob_dropout = tf.placeholder(tf.float32, name='dropout')
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Embed and encode sequences
        sequences_embedded = tf.nn.embedding_lookup(self.word_embeddings, sequences)
        encoded_sequences = self.encoder(sequences_embedded, sequences_lengths, keep_prob_dropout)

        # Take last hidden state of GRU and put them through a nonlinear and a linear FC layer
        with tf.name_scope('non_linear_layer'):
            encoded_sentences_BN = self.batch_norm_wrapper(encoded_sequences, self.is_train)
            non_linear = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(encoded_sentences_BN, 64)),
                                       keep_prob=keep_prob_dropout)

        with tf.name_scope('final_layer'):
            non_linear_BN = self.batch_norm_wrapper(non_linear, self.is_train)
            logits = tf.contrib.layers.linear(non_linear_BN, 4)

        # Compute mean loss on this batch, consisting of cross entropy loss and L2 loss
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        # Perform training operation
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 100, 0.96, staircase=True)
        opt_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=global_step, learning_rate=learning_rate,
                                                 optimizer='Adam', clip_gradients=2.0,
                                                 learning_rate_decay_fn=None, summaries=None)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)

        probs = tf.nn.softmax(logits)
        with tf.name_scope('accuracy'):
            pred = tf.argmax(logits, 1)
            correct_prediction = tf.equal(labels, pred)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        if self.is_train:
            with tf.Session() as session:
                train_loss_writer = tf.summary.FileWriter(str(self.path + 'tensorboard/train_loss'), session.graph)
                train_summary_writer = tf.summary.FileWriter(str(self.path + 'tensorboard/train_summary'),
                                                             session.graph)
                val_summary_writer = tf.summary.FileWriter(str(self.path + 'tensorboard/val_summary'), session.graph)

                embedding_writer = tf.summary.FileWriter(str(self.path + 'tensorboard/'), session.graph)

                config = projector.ProjectorConfig()
                embedding = config.embeddings.add()
                embedding.tensor_name = self.word_embeddings.name

                projector.visualize_embeddings(embedding_writer, config)

                merged = tf.summary.merge_all()
                tf.global_variables_initializer().run()

                start_time = time.time()
                batch_times = []
                n = self.train_data.shape[0]
                print('Start training')
                try:
                    for epoch in range(self.num_epochs):
                        self.is_train = True
                        epoch_time = time.time()
                        print('----- Epoch', epoch, '-----')

                        perm = np.random.permutation(len(self.train_data))
                        self.train_data_perm = self.train_data[perm]
                        self.train_labels_perm = self.train_labels[perm]
                        self.train_lengths_perm = self.train_lengths[perm]

                        total_loss = 0

                        for i in range(n // self.batch_size):
                            batch_start = time.time()
                            batch_data = self.train_data_perm[i * self.batch_size: (i + 1) * self.batch_size]
                            batch_lengths = self.train_lengths_perm[i * self.batch_size: (i + 1) * self.batch_size]
                            batch_labels = self.train_labels_perm[i * self.batch_size: (i + 1) * self.batch_size]

                            train_dict = {sequences: batch_data,
                                          sequences_lengths: batch_lengths,
                                          labels: batch_labels,
                                          keep_prob_dropout: self.keep_prob_dropout}

                            _, batch_loss, batch_accuracy, batch_summary = session.run([opt_op, loss, accuracy, merged],
                                                                                       feed_dict=train_dict)
                            total_loss += batch_loss
                            batch_times.append(time.time() - batch_start)

                            train_loss_writer.add_summary(batch_summary, i + (n // self.batch_size) * epoch)

                            if i % 10 == 0 and i > 0:
                                # Print loss every 10 batches
                                time_per_epoch = np.mean(batch_times) * (n // self.batch_size)
                                remaining_time = int(time_per_epoch - time.time() + epoch_time)
                                string_out = '\rEnd of batch ' + str(i) + '    Train loss:   ' + str(
                                    total_loss / (i * self.batch_size)) + '    Accuracy:   ' + str(batch_accuracy)
                                string_out += '  Elapsed training time : ' + str(int(time.time() - start_time)) + "s, "
                                string_out += str(remaining_time) + "s remaining for this epoch"
                                string_out += '  (' + str(time_per_epoch * 100 / 60 // 1 / 100) + ' min/epoch)'
                                stdout.write(string_out)

                        train_dict = {sequences: self.train_data_perm[:1000],
                                      sequences_lengths: self.train_lengths_perm[:1000],
                                      labels: self.train_labels_perm[:1000],
                                      keep_prob_dropout: 1.0}

                        train_summary, train_loss, train_accuracy = session.run([merged, loss, accuracy],
                                                                                feed_dict=train_dict)
                        train_summary_writer.add_summary(train_summary, epoch)
                        print('\nEpoch train loss: ', train_loss, 'Epoch train accuracy: ', train_accuracy)

                        val_dict = {sequences: self.val_data,
                                    sequences_lengths: self.val_lengths,
                                    labels: self.val_labels,
                                    keep_prob_dropout: 1.0}
                        val_summary, val_loss, val_accuracy = session.run([merged, loss, accuracy], feed_dict=val_dict)
                        val_summary_writer.add_summary(val_summary, epoch)
                        print('\nEpoch val loss: ', val_loss, 'Epoch val accuracy: ', val_accuracy)

                        self.save_model(session, epoch)

                        saver = tf.train.Saver()
                        saver.save(session, os.path.join(self.path + '/tensorboard/', 'model.ckpt'))

                except KeyboardInterrupt:
                    save = input('save?')
                    if 'y' in save:
                        self.save_model(session, epoch)

        elif self.mode == 'Test':
            with tf.Session() as session:
                print('Restoring model...')
                saver = tf.train.Saver()
                saver.restore(session, self.path + 'epoch_19.checkpoint')
                print('Model restored!')

                with open('test_data.pkl', 'rb') as f:
                    test_sequences = pkl.load(f)

                _, _, data_lengths, _, enc_sequences = build_dictionary(test_sequences, vocab=dictionary)
                print(len(data_lengths), np.max(data_lengths), np.min(data_lengths), np.mean(data_lengths),
                      np.std(data_lengths))

                test_dict = {sequences: enc_sequences,
                             sequences_lengths: data_lengths,
                             keep_prob_dropout: 1.0}

                self.probs, self.pred = session.run([probs, pred], feed_dict=test_dict)
                result = pd.DataFrame(np.concatenate((self.probs, np.expand_dims(self.pred, 1)), 1))
                result.columns = pd.Index(['cyto', 'secreted', 'mito', 'nucleus', 'prediction'])
                print(result)


if __name__ == '__main__':
    tf.reset_default_graph()

    labels_string = ['cyto', 'secreted', 'mito', 'nucleus']

    model = ProteinClassification(mode='Train',
                                  path='./model/',
                                  folds=5,
                                  embedding_size=32,
                                  hidden_size=64,
                                  batch_size=32,
                                  keep_prob_dropout=0.7,
                                  learning_rate=0.01,
                                  num_epochs=20)

    model.run()
