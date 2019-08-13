import os
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import numpy as np


def save_json(data, json_path):
    with open(json_path, 'w') as outfile:
        json.dump(data, outfile)


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def plotter(training_data, folder):

    epochs = [x[0] for x in training_data]

    mean_loss = [x[1] for x in training_data]
    mean_acc = [x[2] for x in training_data]
    mean_f1 = [x[3] for x in training_data]

    mean_loss_val = [x[4] for x in training_data]
    mean_acc_val = [x[5] for x in training_data]
    mean_f1_val = [x[6] for x in training_data]

    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(epochs, mean_loss, label='Training Loss')
    ax1.plot(epochs, mean_loss_val, label='Validation Loss')

    ax1.set_title("Training metric 1: Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    plt.savefig(os.path.join(folder, 'loss_vs_epochs.png'))
    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))

    ax2.plot(epochs, mean_acc, label='Training Accuracy')
    ax2.plot(epochs, mean_acc_val, label = 'Validation Accuracy')
    ax2.set_title("Training metric 2: Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.savefig(os.path.join(folder, 'accuracy_vs_epochs.png'))
    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))

    ax3.plot(epochs, mean_f1, label='Training F1 Score')
    ax3.plot(epochs, mean_f1_val, label='Validation F1 Score')
    ax3.set_title("Training metric 3: F1 Score")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("F1 Score")
    ax3.legend()

    plt.savefig(os.path.join(folder, 'f1_vs_epochs.png'))
    plt.clf()

def use_only_specified_gpu(gpu_id):

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)

def build_attention_network(network, batch_size, sequence_len, units_first_lstm, model_type, feature_map_size, image_input_size, filter_no):
    """
    Build the attention network.
    sample_sequence - mock data structure (batch_size, sequence_len, W, H, C)
    sample_init_input - mock data structure (batch_size, sequence_len, K, K, D)

    """
    sample_sequence = np.random.rand(batch_size, sequence_len, image_input_size, image_input_size, 3).astype(np.float32)


    if model_type == 'ALSTM':
        sample_init_input = np.random.rand(batch_size,
                                           sequence_len,
                                           feature_map_size,
                                           feature_map_size,
                                           filter_no).astype(np.float32)

        network.hidden_state_init.build(sample_init_input.shape)
        network.cell_state_init.build(sample_init_input.shape)

        hidden_state = tf.zeros((batch_size, units_first_lstm))
        _ , _, _ = network(sample_sequence[:, 0, :, :, :], hidden_state)

    elif model_type == 'ConvALSTM':
        hidden_state = tf.zeros((batch_size, feature_map_size, feature_map_size, units_first_lstm))
        _ , _, _ = network(sample_sequence[:, 0, :, :, :], hidden_state)

    network.summary()


class AttentionTrainer:

    def __init__(self, network, network_name, optimizer, loss_object, loss_object_attention_pen,
                 train_loss, train_accuracy, train_precision, train_recall,
                 val_loss, val_accuracy, val_precision, val_recall,
                 test_loss, test_accuracy, test_precision, test_recall, batch_size, penalty_coeff, weight_decay):

        self.network = network
        self.network_name = network_name
        self.loss_object = loss_object
        self.loss_object_attention = loss_object_attention_pen
        self.optimizer = optimizer

        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.train_precision = train_precision
        self.train_recall = train_recall

        self.val_loss = val_loss
        self.val_accuracy = val_accuracy
        self.val_precision = val_precision
        self.val_recall = val_recall

        self.test_loss = test_loss
        self.test_accuracy = test_accuracy
        self.test_precision = test_precision
        self.test_recall = test_recall

        self.batch_size = batch_size
        self.penalty_coefficient = penalty_coeff
        self.weight_decay = weight_decay

    def attention_penalty(self,x):
        x = tf.reduce_sum(x, axis=0)
        tensor_ones = tf.ones(shape=(x.shape[0], x.shape[1], x.shape[2]))
        loss = (tensor_ones - x)**2
        loss = tf.reduce_sum(loss, axis = (0, 1, 2))
        return loss

    @tf.function
    def train_step(self, images, labels):
        loss = 0

        with tf.GradientTape() as tape:

            if self.network_name == 'ALSTM':
                batch_feature_cube_sequence = self.network.get_batch_feature_cube_sequence(images)
                hidden_state, cell_state = self.network.reset_hidden_and_cell_state(batch_feature_cube_sequence)
                self.network.lstm1.initial_state = cell_state
                all_attention_weights = tf.zeros((0,
                                                  self.batch_size,
                                                  batch_feature_cube_sequence.shape[2],
                                                  batch_feature_cube_sequence.shape[2]))

                for i in range(0, images.shape[1]):
                    input_image = images[:, i, :, :, :]
                    predictions, hidden_state, attention_weights = self.network(input_image, hidden_state)

                    self.train_accuracy.update_state(labels, predictions)

                    loss = loss + self.loss_object(labels, predictions)
                    attention_weights = tf.expand_dims(attention_weights, 0)
                    all_attention_weights = tf.concat([all_attention_weights, attention_weights], 0)

                classification_loss = loss / int(images.shape[1])
                attention_loss = self.attention_penalty(all_attention_weights)/ int(images.shape[1])*self.batch_size
                regularization_term = tf.add_n([tf.nn.l2_loss(v) for v in self.network.trainable_variables if 'bias' not in v.name])

                total_loss = classification_loss + self.penalty_coefficient*attention_loss + self.weight_decay*regularization_term

            elif self.network_name == 'ConvALSTM':
                hidden_state = self.network.get_initial_hidden_state(self.batch_size)
                for i in range(0, images.shape[1]):
                    input_image = images[:, i, :, :, :]
                    predictions, hidden_state, attention_weights = self.network(input_image, hidden_state)

                    self.train_accuracy.update_state(labels, predictions)

                    loss = loss + self.loss_object(labels, predictions)

                classification_loss = loss / int(images.shape[1])
                total_loss = classification_loss

        gradients = tape.gradient(total_loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        self.train_loss.update_state(total_loss)
        return total_loss


    @tf.function
    def val_step(self, images, labels):
        loss = 0

        if self.network_name == 'ALSTM':
            batch_feature_cube_sequence = self.network.get_batch_feature_cube_sequence(images)
            hidden_state, cell_state = self.network.reset_hidden_and_cell_state(batch_feature_cube_sequence)
            self.network.lstm1.initial_state = cell_state
            all_attention_weights = tf.zeros((0,
                                              self.batch_size,
                                              batch_feature_cube_sequence.shape[2],
                                              batch_feature_cube_sequence.shape[2]))

            for i in range(0, images.shape[1]):
                input_image = images[:, i, :, :, :]
                predictions, hidden_state, attention_weights = self.network(input_image, hidden_state)

                self.val_accuracy.update_state(labels, predictions)

                loss = loss + self.loss_object(labels, predictions)
                attention_weights = tf.expand_dims(attention_weights, 0)
                all_attention_weights = tf.concat([all_attention_weights, attention_weights], 0)


            classification_loss = loss / int(images.shape[1])
            attention_loss = self.attention_penalty(all_attention_weights)/ int(images.shape[1]) * self.batch_size
            regularization_term = tf.add_n([tf.nn.l2_loss(v) for v in self.network.trainable_variables if 'bias' not in v.name])
            total_loss = classification_loss + self.penalty_coefficient * attention_loss + self.weight_decay*regularization_term

        elif self.network_name == 'ConvALSTM':
            hidden_state = self.network.get_initial_hidden_state(self.batch_size)
            for i in range(0, images.shape[1]):
                input_image = images[:, i, :, :, :]
                predictions, hidden_state, attention_weights = self.network(input_image, hidden_state)

                self.val_accuracy.update_state(labels, predictions)

                loss = loss + self.loss_object(labels, predictions)

            classification_loss = loss / int(images.shape[1])
            total_loss = classification_loss

        self.val_loss.update_state(total_loss)

    def save_weights(self, epoch, path):
        self.network.save_weights(os.path.join(path, '{}_{}.h5'.format(self.network.name, epoch)))










