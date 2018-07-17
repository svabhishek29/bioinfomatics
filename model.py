import operator
import os
import pickle as pkl
from copy import copy
from sys import stdout

import numpy as np
import pandas as pd
import tensorflow as tf
# from util import encode_labels
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt

# from util import get_data
from util import build_dictionary


class skipthought(object):

    def __init__(self, mode, path, folds, embedding_size, hidden_size, hidden_layers, batch_size, keep_prob_dropout, L2,
                 learning_rate, val_size, bidirectional, mask, num_epochs=100):

        self.mode = mode
        self.path = path
        self.folds = folds
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.keep_prob_dropout = keep_prob_dropout
        self.L2 = L2
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.bidirectional = bidirectional
        self.num_epochs = num_epochs
        self.mask = mask

    def save_model(self, session, epoch):

        '''
        Helper function to save the TF graph
        '''

        if not os.path.exists('./model/'):
            os.mkdir('./model/')
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        if not os.path.exists('./model/epoch_%d.checkpoint' % epoch):
            saver.save(session, './model/epoch_%d.checkpoint' % epoch)
        else:
            saver.save(session, './model/epoch_%d.checkpoint' % epoch)

    def embed_data(self, data):

        '''
        Takes a batch of amino acids as input and embeds them using the current embedding matrix.
        '''

        return tf.nn.embedding_lookup(self.word_embeddings, data)

    def encoder(self, sentences_embedded, sentences_lengths, dropout, bidirectional=False):

        '''
        This functions uses a GRU cell to encode amino acid sequences

        Takes as inputs 
        -   embedded amino acid sequences
        -   the corresponding sequence lengths
        -   the dropout keep-probability and 
        -   a flag for whether a one-directional or bidirectional model shall be trained
    
        Returns the last memory state of the GRU cell
        '''

        with tf.variable_scope("encoder") as varscope:
            cell = tf.contrib.rnn.GRUCell(self.hidden_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * self.hidden_layers, state_is_tuple=True)

            if bidirectional:
                print('Training bidirectional RNN')
                sentences_outputs, sentences_states = tf.nn.bidirectional_dynamic_rnn(cell, cell,
                                                                                      inputs=sentences_embedded,
                                                                                      sequence_length=sentences_lengths,
                                                                                      dtype=tf.float32)
                states_fw, states_bw = sentences_states
                sentences_states_h = tf.concat([states_fw[-1], states_bw[-1]], axis=1)
                print(sentences_states)
                print(states_fw)
                print(sentences_states_h)

            else:
                print('Training one-directional RNN')
                sentences_outputs, sentences_states = tf.nn.dynamic_rnn(cell,
                                                                        inputs=sentences_embedded,
                                                                        sequence_length=sentences_lengths,
                                                                        dtype=tf.float32)
                sentences_states_h = sentences_states[-1]
                print(sentences_states_h)

        return sentences_states_h

    def get_CE_loss(self, labels, logits):

        '''
        Takes as inputs:
        -   true labels for each amino acid sequence in the batch
        -   predicted logits for each amino acid sequence in the batch

        Returns the mean cross entropy loss for this batch
        '''

        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    def get_L2_loss(self):

        '''
        Returns the L2 loss from all parameters of the model (excluding biases)
        '''

        all_vars = tf.trainable_variables()
        return tf.add_n([tf.nn.l2_loss(v) for v in all_vars if 'bias' not in v.name]) * self.L2

    def batch_norm_wrapper(self, inputs, is_training, decay=0.999):

        '''
        Takes as input the inputs that go into a layer of the network.
        Returns the normalised (with respect to batch mean and variance) inputs
        Adopted from: http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
        '''

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

    def summary_stats(self, lengths, labels, name):

        '''
        Takes as input the lengths and labels of amino acid sequences
        Prints and returns pandas dataframes containing descriptive statistics
        '''

        bins = [0, 100, 500, 1000, 1500, 1999]
        labels_string = ['cyto', 'secreted', 'mito', 'nucleus']

        df = pd.DataFrame({'length': lengths, 'label': labels})
        table = pd.crosstab(np.digitize(df.length, bins), df.label)
        table.index = pd.Index(['[0, 100)', '[100, 500)', '[500, 1000]', '[1000, 1500)', '[1500, 2000)', '[2000, inf]'],
                               name="Bin")
        table.columns = pd.Index(labels_string, name="Class")

        sum_row = {col: table[col].sum() for col in table}
        sum_df = pd.DataFrame(sum_row, index=["Total"])
        table = table.append(sum_df)
        table['Total'] = table.sum(axis=1)

        print('\n~~~~~~~ Summary stats for %s set ~~~~~~~' % name)
        print('\nCount of sequence lengths by class')
        print(table)
        print('\nDescriptive statistics')
        print(df.describe())

        return df, table

    def confusion(self, gold, prediction, lengths, min_length=0, max_length=np.inf):

        '''
        Takes as input the gold and predicted labels
        Returns a pandas dataframe containing a confusion matrix
        '''

        labels_string = ['cyto', 'secreted', 'mito', 'nucleus']
        a = lengths > min_length
        b = lengths < max_length
        mask = a * b
        y_hat = pd.Series(prediction[mask], name='Predicted')
        y = pd.Series(gold[mask], name='Actual')
        df_confusion = pd.crosstab(y, y_hat)
        sum_row = {col: df_confusion[col].sum() for col in df_confusion}
        sum_df = pd.DataFrame(sum_row, index=["Total"])
        df_confusion = df_confusion.append(sum_df)
        df_confusion['Total'] = df_confusion.sum(axis=1)

        # df_confusion.index = pd.Index(labels_string)
        # df_confusion.rows = pd.Index(labels_string)

        return df_confusion

    def run(self):

        '''
        Runs the model according to the specified settings
        -   If mode = Train: Train a GRU model using the training data
        -   If mode = Val: Load the saved GRU model and evaluate it on the validation fold
        -   If mode = Test: Load the saved GRU model and evaluate it on the blind test set
        '''

        self.is_train = (self.mode == 'Train')

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        # Load the training data
        with open('train_data.pkl', 'rb') as f:
            data_sequences = pkl.load(f)
        with open('train_labels.pkl', 'rb') as f:
            data_labels = pkl.load(f)

        dictionary, reverse_dictionary, data_lengths, self.max_seq_len, enc_sequences = build_dictionary(data_sequences)
        self.dictionary = sorted(dictionary.items(), key=operator.itemgetter(1))
        print(self.dictionary)
        self.vocabulary_size = len(dictionary)
        self.val_size = len(data_sequences) // self.folds
        fold = self.mask
        print('Training fold number %d. Each fold of size %d' % (fold, len(data_sequences) // self.folds))

        # Truncates sequences at length 2000 and returns descriptive statistics.
        # This is done by concatenating the first 1900 and the last 100 amino acids.

        if self.is_train:
            self.max_seq_len = 2000
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

            self.train_statistics, self.train_frame = self.summary_stats(self.train_lengths, self.train_labels, 'train')
            if self.folds == 1:
                self.val_statistics = np.array([])
                self.val_frame = np.array([])
                self.val_original_lengths = np.array([])
            else:
                self.val_statistics, self.val_frame = self.summary_stats(self.val_lengths, self.val_labels,
                                                                         'validation')

            this_data = [self.train_data,
                         self.train_labels,
                         self.train_lengths,
                         self.val_data,
                         self.val_labels,
                         self.val_lengths,
                         self.train_statistics,
                         self.train_frame,
                         self.val_statistics,
                         self.val_frame,
                         self.train_original_lengths,
                         self.val_original_lengths
                         ]

            with open(self.path + 'this_data.pkl', 'wb') as f:
                pkl.dump(this_data, f)

        else:
            with open(self.path + 'this_data.pkl', 'rb') as f:
                self.train_data, self.train_labels, self.train_lengths, self.val_data, self.val_labels, self.val_lengths, self.train_statistics, self.train_frame, self.val_statistics, self.val_frame, self.train_original_lengths, self.val_original_lengths = pkl.load(
                    f)

        # Now construct the Tensorflow graph
        print('\r~~~~~~~ Building model ~~~~~~~\r')

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
        sequences_embedded = self.embed_data(sequences)
        encoded_sequences = self.encoder(sequences_embedded, sequences_lengths, keep_prob_dropout,
                                         bidirectional=self.bidirectional)

        # Take last hidden state of GRU and put them through a nonlinear and a linear FC layer
        with tf.name_scope('non_linear_layer'):
            encoded_sentences_BN = self.batch_norm_wrapper(encoded_sequences, self.is_train)
            non_linear = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(encoded_sentences_BN, 64)),
                                       keep_prob=keep_prob_dropout)

        with tf.name_scope('final_layer'):
            non_linear_BN = self.batch_norm_wrapper(non_linear, self.is_train)
            logits = tf.contrib.layers.linear(non_linear_BN, 4)

        # Compute mean loss on this batch, consisting of cross entropy loss and L2 loss
        CE_loss = self.get_CE_loss(labels, logits)
        L2_loss = self.get_L2_loss()
        loss = CE_loss + L2_loss

        # Perform training operation
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 100, 0.96, staircase=True)
        opt_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=global_step, learning_rate=learning_rate,
                                                 optimizer='Adam', clip_gradients=2.0,
                                                 learning_rate_decay_fn=None, summaries=None)

        # Define scalars for Tensorboard
        tf.summary.scalar('CE_loss', CE_loss)
        tf.summary.scalar('L2_loss', L2_loss)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)

        # Compute accuracy of prediction
        probs = tf.nn.softmax(logits)
        with tf.name_scope('accuracy'):
            pred = tf.argmax(logits, 1)
            correct_prediction = tf.equal(labels, pred)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        # If in training mode:
        # - shuffle data set before each epoch
        # - train model using mini batches
        # - track performance on train and validation set throughout training

        if self.is_train == True:
            with tf.Session() as session:
                train_loss_writer = tf.summary.FileWriter(str(self.path + 'tensorboard/train_loss'), session.graph)
                train_summary_writer = tf.summary.FileWriter(str(self.path + 'tensorboard/train_summary'),
                                                             session.graph)
                val_summary_writer = tf.summary.FileWriter(str(self.path + 'tensorboard/val_summary'), session.graph)

                # Use the same LOG_DIR where you stored your checkpoint.
                embedding_writer = tf.summary.FileWriter(str(self.path + 'tensorboard/'), session.graph)

                config = projector.ProjectorConfig()
                embedding = config.embeddings.add()
                embedding.tensor_name = self.word_embeddings.name
                # Link this tensor to its metadata file (e.g. labels).
                embedding.metadata_path = os.path.join('./metadata.tsv')

                # Saves a configuration file that TensorBoard will read during startup.
                projector.visualize_embeddings(embedding_writer, config)

                merged = tf.summary.merge_all()
                print('\r~~~~~~~ Initializing variables ~~~~~~~\r')
                tf.global_variables_initializer().run()

                start_time = time.time()
                min_train_loss = np.inf
                batch_times = []
                n = self.train_data.shape[0]
                print('\r~~~~~~~ Starting training ~~~~~~~\r')
                try:
                    train_summaryIndex = -1

                    for epoch in range(self.num_epochs):
                        self.is_train = True
                        epoch_time = time.time()
                        print('----- Epoch', epoch, '-----')
                        print('Shuffling dataset')

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

                        # Train accuracy
                        train_dict = {sequences: self.train_data_perm[:1000],
                                      sequences_lengths: self.train_lengths_perm[:1000],
                                      labels: self.train_labels_perm[:1000],
                                      keep_prob_dropout: 1.0}

                        train_summary, train_loss, train_accuracy = session.run([merged, loss, accuracy],
                                                                                feed_dict=train_dict)
                        train_summary_writer.add_summary(train_summary, epoch)
                        print('\nEpoch train loss: ', train_loss, 'Epoch train accuracy: ', train_accuracy)

                        # Val accuracy
                        val_dict = {sequences: self.val_data,
                                    sequences_lengths: self.val_lengths,
                                    labels: self.val_labels,
                                    keep_prob_dropout: 1.0}
                        val_summary, val_loss, val_accuracy = session.run([merged, loss, accuracy], feed_dict=val_dict)
                        val_summary_writer.add_summary(val_summary, epoch)
                        print('\nEpoch val loss: ', val_loss, 'Epoch val accuracy: ', val_accuracy)

                        self.save_model(session, epoch)

                        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
                        saver.save(session, os.path.join(self.path + '/tensorboard/', 'model.ckpt'))


                except KeyboardInterrupt:
                    save = input('save?')
                    if 'y' in save:
                        self.save_model(session, epoch)

        # If in validation mode:
        # - Load saved model and evaluate on validation fold
        # - Return list containing confusion matrices, and accuracy measures such as FPR and TPR

        elif self.mode == 'Val':
            with tf.Session() as session:
                print('Restoring model...')
                saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
                saver.restore(session, self.path + 'tensorboard/model.ckpt')
                print('Model restored!')

                val_dict = {sequences: self.val_data,
                            sequences_lengths: self.val_lengths,
                            labels: self.val_labels,
                            keep_prob_dropout: 1.0}

                self.val_pred, self.val_accuracy, self.val_probs = session.run([pred, accuracy, probs],
                                                                               feed_dict=val_dict)

                _ = self.summary_stats(self.val_lengths, self.val_labels, 'val')

                print('\nConfusion matrix (all sequence lengths):')
                val_confusion_1 = self.confusion(gold=self.val_labels, prediction=self.val_pred,
                                                 lengths=self.val_original_lengths, min_length=0, max_length=np.inf)
                print(val_confusion_1)

                print('\nConfusion matrix (sequence length < 2000):')
                val_confusion_2 = self.confusion(gold=self.val_labels, prediction=self.val_pred,
                                                 lengths=self.val_original_lengths, min_length=0, max_length=2000)
                print(val_confusion_2)

                print('\nConfusion matrix (sequence length > 2000):')
                val_confusion_3 = self.confusion(gold=self.val_labels, prediction=self.val_pred,
                                                 lengths=self.val_original_lengths, min_length=2000, max_length=np.inf)
                print(val_confusion_3)

                print('\n Val accuracy:', self.val_accuracy)
                print('\n Val accuracy when length <2000:',
                      np.sum((self.val_pred == self.val_labels) * (self.val_original_lengths <= 2000)) / np.sum(
                          self.val_original_lengths <= 2000))
                print('\n Val accuracy when length >2000:',
                      np.sum((self.val_pred == self.val_labels) * (self.val_original_lengths > 2000)) / np.sum(
                          self.val_original_lengths > 2000))

                this_sum = np.zeros([3, 5])
                this_auc = np.zeros([1, 5])
                this_TPR = []
                this_FPR = []

                total_tp = 0
                total_fp = 0
                total_fn = 0
                total_tn = 0

                for i in range(4):
                    tp = np.sum((self.val_labels == i) * (self.val_pred == i))
                    fp = np.sum((self.val_labels != i) * (self.val_pred == i))
                    fn = np.sum((self.val_labels == i) * (self.val_pred != i))
                    tn = np.sum((self.val_labels != i) * (self.val_pred != i))

                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    total_tn += tn
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * prec * recall / (prec + recall) if prec * recall > 0 else 0.0
                    this_sum[:, i] = np.array([prec, recall, f1])
                    this_auc[:, i] = roc_auc_score(self.val_labels == i, self.val_pred == i)
                    if i < 4:
                        this_FPR.append(roc_curve(self.val_labels == i, self.val_probs[:, i])[0])
                        this_TPR.append(roc_curve(self.val_labels == i, self.val_probs[:, i])[1])

                prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                f1 = 2 * prec * recall / (prec + recall) if prec * recall > 0 else 0.0
                this_sum[:, 4] = np.array([prec, recall, f1])
                this_sum = np.concatenate((this_sum, this_auc), 0)

                self.this_sum = pd.DataFrame(this_sum)
                self.this_sum.index = pd.Index(['Precision', 'Recall', 'F1', 'AUC'])
                self.this_sum.columns = pd.Index(['cyto', 'secreted', 'mito', 'nucleus', 'Total'])

                print(self.this_sum)

                if self.is_train == False:
                    return [val_confusion_1, val_confusion_2, val_confusion_3, self.this_sum, this_FPR, this_TPR]

        # If in test model:
        # - Load saved model and evaluate on test set
        # - Print predicted probabilities for each protein in the test set

        elif self.mode == 'Test':
            with tf.Session() as session:
                print('Restoring model...')
                saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
                saver.restore(session, self.path + 'model.checkpoint')
                print('Model restored!')

                with open('test_data.pkl', 'rb') as f:
                    test_sequences = pkl.load(f)
                with open('test_labels.pkl', 'rb') as f:
                    test_labels = pkl.load(f)

                _, _, data_lengths, _, enc_sequences = build_dictionary(test_sequences, vocab=dictionary)

                test_dict = {sequences: enc_sequences,
                             sequences_lengths: data_lengths,
                             keep_prob_dropout: 1.0}

                self.probs, self.pred = session.run([probs, pred], feed_dict=test_dict)
                result = pd.DataFrame(np.concatenate((self.probs, np.expand_dims(self.pred, 1)), 1))
                result.columns = pd.Index(['cyto', 'secreted', 'mito', 'nucleus', 'prediction'])
                print(result)


def stats():
    '''
    Helper function to print descriptive statistics of training data
    '''

    with open('train_data.pkl', 'rb') as f:
        data_sequences = pkl.load(f)
    with open('train_labels.pkl', 'rb') as f:
        labels = pkl.load(f)

    _, _, lengths, _, _ = build_dictionary(data_sequences)

    bins = [0, 100, 500, 1000, 1500, 1999]
    labels_string = ['cyto', 'secreted', 'mito', 'nucleus']

    df = pd.DataFrame({'length': lengths, 'label': labels})
    table = pd.crosstab(np.digitize(df.length, bins), df.label)

    table.index = pd.Index(['[0, 100)', '[100, 500)', '[500, 1000]', '[1000, 1500)', '[1500, 2000)', '[2000, inf]'],
                           name="Bin")
    table.columns = pd.Index(labels_string, name="Class")

    sum_row = {col: table[col].sum() for col in table}
    sum_df = pd.DataFrame(sum_row, index=["Total"])
    table = table.append(sum_df)
    table['Total'] = table.sum(axis=1)

    print('\n~~~~~~~ Summary stats for %s set ~~~~~~~')
    print('\nCount of sequence lengths by class')
    print(table)
    print('\nDescriptive statistics')
    print(df.describe())


if __name__ == '__main__':

    labels_string = ['cyto', 'secreted', 'mito', 'nucleus']

    for i in range(1, 6):
        tf.reset_default_graph()

        model = skipthought(mode='Train',
                            path=str('./model_%d/' % i),
                            folds=5,
                            embedding_size=32,
                            hidden_size=128,
                            hidden_layers=1,
                            batch_size=32,
                            keep_prob_dropout=0.7,
                            L2=0.0,
                            learning_rate=0.01,
                            val_size=1700,
                            bidirectional=False,
                            mask=i,
                            num_epochs=20)

        model.run()

    # Obtain cross-validated confusion matrices
    # outputs = []
    # for i in range(1, 6):
    #     tf.reset_default_graph()
    #     print('\r%d' % i)
    #     this_path = str('./model_%d/' % i)
    #     model = skipthought(mode='Val',
    #                         path=this_path,
    #                         folds=5,
    #                         embedding_size=32,
    #                         hidden_size=64,
    #                         hidden_layers=1,
    #                         batch_size=32,
    #                         keep_prob_dropout=0.7,
    #                         L2=0.0,
    #                         learning_rate=0.01,
    #                         val_size=1700,
    #                         bidirectional=False,
    #                         mask=i,
    #                         num_epochs=20)
    #
    #     outputs.append(model.run())
    #     print('\n\n\n')
    #
    # val_confusion_1 = (outputs[0][0] + outputs[1][0] + outputs[2][0] + outputs[3][0] + outputs[4][0]) / 5
    # summary = (outputs[0][3] + outputs[1][3] + outputs[2][3] + outputs[3][3] + outputs[4][3]) / 5
    #
    # print("mean confusion matrix")
    # print(val_confusion_1)
    # print("mean summary")
    # print(summary)
    #
    # for i in range(4):
    #     plt.close()
    #     plt.plot(outputs[0][4][i], outputs[0][5][i])
    #     plt.plot(outputs[1][4][i], outputs[1][5][i])
    #     plt.plot(outputs[2][4][i], outputs[2][5][i])
    #     plt.plot(outputs[3][4][i], outputs[3][5][i])
    #     plt.plot(outputs[4][4][i], outputs[4][5][i])
    #     plt.xlabel('FPR')
    #     plt.ylabel('TPR')
    #     name = labels_string[i]
    #     plt.savefig('./plots/%s.png' % name)
    #
    # stats()
