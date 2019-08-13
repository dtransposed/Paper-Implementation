from tensorflow.keras.layers import Dense, LSTM, ConvLSTM2D, Flatten, Dropout, Conv2D, GlobalMaxPooling2D
from tensorflow.keras import Model
import tensorflow as tf


class Encoder(Model):

    def __init__(self, base_model, train_base):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.train_base = train_base
        for layer in self.base_model.layers:
            layer.trainable = self.train_base

    def call(self, x):
        return self.base_model(x)


class HiddenStateInit(Model):

    def __init__(self, output_dim, use_dropout):
        super(HiddenStateInit, self).__init__()
        self.mlp_h = Dense(output_dim, activation='tanh')
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout1 = Dropout(0.5)

    def call(self, x):
        if self.use_dropout:
            x = self.mlp_h(self.dropout1(x))
        else:
            x = self.mlp_h(x)
        return x


class CellStateInit(Model):

    def __init__(self, output_dim, use_dropout):
        super(CellStateInit, self).__init__()
        self.mlp_c = Dense(output_dim, activation='tanh')
        self.use_dropout = use_dropout

        if self.use_dropout:
            self.dropout1 = Dropout(0.5)

    def call(self, x):
        if self.use_dropout:
            x = self.mlp_c(self.dropout1(x))
        else:
            x = self.mlp_c(x)
        return x


class BahdanauAttention(Model):

    def __init__(self, feature_map_size, units, use_dropout):
        super(BahdanauAttention, self).__init__()
        self.feature_map_size = feature_map_size
        self.units = units
        self.mlp1 = Dense(self.units/2)
        self.mlp2 = Dense(self.units/2)
        self.mlp3 = Dense(self.feature_map_size**2)
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout1 = Dropout(0.5)
            self.dropout2 = Dropout(0.5)
            self.dropout3 = Dropout(0.5)
        self.flatten = Flatten()

    def call(self, x, h):
        if self.use_dropout:
            scores = self.mlp3(self.dropout3(tf.nn.tanh(self.mlp1(self.dropout1(h)) + self.mlp2(self.dropout2(tf.reduce_mean(x, axis=(1, 2)))))))
        else:
            scores = self.mlp3(tf.nn.tanh(self.mlp1(h) + self.mlp2(tf.reduce_mean(x, axis=(1, 2)))))

        attention_map = tf.nn.softmax(scores, axis=1)
        attention_map = tf.reshape(attention_map, (attention_map.shape[0], self.feature_map_size, self.feature_map_size,1))

        x = x * attention_map
        x = tf.reduce_sum(x, axis=(1, 2))

        return x, attention_map


class ConvBahdanauAttention(Model):

    def __init__(self, feature_map_size, units):
        super(ConvBahdanauAttention, self).__init__()
        self.feature_map_size = feature_map_size
        self.units = units
        self.conv1 = Conv2D(filters=units, kernel_size=1)
        self.conv2 = Conv2D(filters=units, kernel_size=1)
        self.conv3 = Conv2D(filters=1, kernel_size=1)

    def call(self,x, h):

        scores = self.conv3(tf.nn.tanh(self.conv2(x) + self.conv1(h)))
        attention_map_flatten = tf.reshape(scores, (scores.shape[0], scores.shape[1] * scores.shape[2], scores.shape[3]))
        attention_map_flatten = tf.nn.softmax(attention_map_flatten, axis=1)
        attention_map = tf.reshape(attention_map_flatten, scores.shape)

        x = x * attention_map

        return x, attention_map


##############################
#          ALSTM
#        (Temporal)
##############################

class ALSTM(Model):

    def __init__(self, base_model, use_dropout, train_base, units_first_lstm, no_classes, feature_map_size):
        super(ALSTM, self).__init__()

        self.units_first_lstm = units_first_lstm
        self.no_classes = no_classes
        self.use_dropout = use_dropout
        self.no_classes = no_classes
        self.feature_map_size = feature_map_size

        self.hidden_state_init = HiddenStateInit(self.units_first_lstm, self.use_dropout)
        self.cell_state_init = CellStateInit(self.units_first_lstm, self.use_dropout)

        self.encoder = Encoder(base_model, train_base)
        self.attention = BahdanauAttention(self.feature_map_size, self.units_first_lstm, self.use_dropout)

        self.lstm1 = LSTM(units = self.units_first_lstm,
                          return_sequences=False,
                          return_state=False,
                          stateful=False,
                          recurrent_initializer='glorot_uniform',
                          trainable=True)

        if self.use_dropout == True:
            self.dropout = Dropout(0.5)

        self.d1 = Dense(self.no_classes, activation='softmax',trainable=True)

    def call(self, x, h):

        x = self.encoder(x)
        initial_batch_size, filter_number = x.shape[0], x.shape[3]
        x, attention_map = self.attention(x, h)
        assert x.shape == (initial_batch_size, filter_number)

        x = tf.expand_dims(x, axis = 1)
        state_h = self.lstm1(x)
        x = state_h
        if self.use_dropout:
            x = self.dropout(x)
        x = self.d1(x)

        attention_map = tf.reshape( attention_map, (attention_map.shape[0], attention_map.shape[1], attention_map.shape[1]))

        assert x.shape == (initial_batch_size, self.no_classes)
        assert attention_map.shape == (initial_batch_size, self.feature_map_size, self.feature_map_size)

        return x, state_h, attention_map

    def get_batch_feature_cube_sequence(self, x):
        """
        Pass tensor of shape (batch_size, sequence_len, W, H, C=3)
        through the encoder
        """
        batch_size, sequence_len, W, H, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]

        x = tf.reshape(x, shape = (batch_size * sequence_len, W, H, C))
        x = self.encoder(x)
        filter_number = x.shape[3]
        x = tf.reshape(x, shape=(batch_size, sequence_len, x.shape[1], x.shape[2], x.shape[3]))

        assert x.shape == (batch_size, sequence_len, self.feature_map_size, self.feature_map_size, filter_number)

        return x

    def reset_hidden_and_cell_state(self, batch_feature_cube_sequence):
        '''
        Initialize hidden_state and cell state from
        batch_feature_cube_sequence of shape
        (batch_size, sequence_len, K, K, D)
        '''

        compressed_representation = tf.math.reduce_mean(batch_feature_cube_sequence, axis = (1,2,3))

        assert compressed_representation.shape == (batch_feature_cube_sequence.shape[0],
                                                   batch_feature_cube_sequence.shape[4])

        hidden_state_0 = self.hidden_state_init(compressed_representation)
        cell_state_0 = self.cell_state_init(compressed_representation)

        assert hidden_state_0.shape == (batch_feature_cube_sequence.shape[0],
                                        self.units_first_lstm)
        assert cell_state_0.shape == (batch_feature_cube_sequence.shape[0],
                                      self.units_first_lstm)

        return hidden_state_0, cell_state_0


##############################
#         ConvALSTM
#        (Temporal)
##############################

class ConvALSTM(Model):

    def __init__(self, base_model, use_dropout, train_base, units_first_lstm, no_classes, feature_map_size):
        super(ConvALSTM, self).__init__()

        self.units_first_lstm = units_first_lstm
        self.no_classes = no_classes
        self.use_dropout = use_dropout
        self.no_classes = no_classes
        self.feature_map_size = feature_map_size

        self.encoder = Encoder(base_model, train_base)
        self.attention = ConvBahdanauAttention(self.feature_map_size, self.units_first_lstm)

        self.lstm1 = ConvLSTM2D (filters=self.units_first_lstm, kernel_size=3, padding='same', return_sequences=False,
                                 return_state= False, stateful=False, recurrent_initializer='glorot_uniform', trainable=True)

        self.flatten = Flatten()
        self.d1 = Dense(self.no_classes, activation = 'softmax',trainable = True)
        if self.use_dropout == True:
            self.dropout = Dropout(0.5)

    def call(self, x, h):

        x = self.encoder(x)
        initial_batch_size, filter_number = x.shape[0], x.shape[3]
        x, attention_map = self.attention(x, h)

        x = tf.expand_dims(x, axis = 1)
        state_h = self.lstm1(x)
        x = state_h
        x = self.flatten(x)

        if self.use_dropout:
            x = self.dropout(x)

        x = self.d1(x)

        attention_map = tf.reshape(attention_map, (attention_map.shape[0], attention_map.shape[1], attention_map.shape[1]))

        assert x.shape == (initial_batch_size, self.no_classes)
        assert attention_map.shape == (initial_batch_size, self.feature_map_size, self.feature_map_size)
        assert state_h.shape == (initial_batch_size, self.feature_map_size, self.feature_map_size, self.units_first_lstm)

        return x, state_h, attention_map

    def get_initial_hidden_state(self, batch_shape):
        return tf.zeros((batch_shape, self.feature_map_size, self.feature_map_size, self.units_first_lstm))

