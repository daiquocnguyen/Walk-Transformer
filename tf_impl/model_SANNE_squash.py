import tensorflow as tf
from tensor2tensor.models import transformer
import math

epsilon = 1e-9

class SANNE(object):
    def __init__(self, sequence_length, num_hidden_layers, vocab_size, feature_dim_size, batch_size, num_heads, ff_hidden_size, initialization, num_sampled, num_neighbors, use_pos):
        # Placeholders for input, output
        self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [batch_size*sequence_length*num_neighbors, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.name_scope("input_feature"):
            if initialization != []:
                self.input_feature = tf.get_variable(name="input_feature_1", initializer=initialization, trainable=False)
            else:
                self.input_feature = tf.get_variable(name="input_feature_2", shape=[vocab_size, feature_dim_size], initializer=tf.contrib.layers.xavier_initializer(seed=1234))
        #Inputs for Transformer Encoder
        self.inputTransfG = tf.nn.embedding_lookup(self.input_feature, self.input_x)
        self.inputTransfG = tf.expand_dims(self.inputTransfG, axis=-1)
        self.inputTransfG = squash(self.inputTransfG)
        self.inputTransfG = tf.reshape(self.inputTransfG, [batch_size, sequence_length, 1, feature_dim_size])

        self.hparams = transformer.transformer_base()
        self.hparams.hidden_size = feature_dim_size
        self.hparams.batch_size = batch_size * sequence_length
        self.hparams.max_length = sequence_length
        self.hparams.num_hidden_layers = num_hidden_layers
        self.hparams.num_heads = num_heads
        self.hparams.filter_size = ff_hidden_size
        self.hparams.use_target_space_embedding = False
        # No positional embedding
        if use_pos == 0:
            self.hparams.pos = None

        #Transformer Encoder
        self.encoder = transformer.TransformerEncoder(self.hparams, mode=tf.estimator.ModeKeys.TRAIN)
        self.outputEncoder = self.encoder({"inputs":self.inputTransfG, "targets": 0, "target_space_id": 0})[0]

        self.outputEncoder = tf.reshape(self.outputEncoder, [batch_size, sequence_length, feature_dim_size, 1])
        self.outputEncoder = squash(self.outputEncoder)

        self.outputEncoder = tf.squeeze(self.outputEncoder)

        self.outputEncoderInd = tf.reshape(self.outputEncoder, [batch_size*sequence_length, feature_dim_size])

        self.outputEncoder = tf.tile(self.outputEncoder, [1, 1, num_neighbors])
        self.outputEncoder = tf.reshape(self.outputEncoder, [batch_size*sequence_length*num_neighbors, feature_dim_size])

        self.outputEncoder = tf.nn.dropout(self.outputEncoder, self.dropout_keep_prob)

        with tf.name_scope("embedding"):
            self.embedding_matrix = tf.get_variable(
                    "W", shape=[vocab_size, feature_dim_size],
                    initializer=tf.contrib.layers.xavier_initializer(seed=1234))

            self.softmax_biases = tf.Variable(tf.zeros([vocab_size]))

        self.total_loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=self.embedding_matrix, biases=self.softmax_biases, inputs=self.outputEncoder,
                                       labels=self.input_y, num_sampled=num_sampled, num_classes=vocab_size))

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
        tf.logging.info('Seting up the main structure')


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)