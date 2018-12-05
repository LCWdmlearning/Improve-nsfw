import math
import numpy as np
import tensorflow as tf
from enum import Enum, unique


@unique
class InputType(Enum):
    TENSOR = 1
    BASE64_JPEG = 2


class OpenNsfwModel:

    def __init__(self):
        self.weights = {}
        self.bn_epsilon = 1e-5  # Default used by Caffe

    def build(self, weights_path="open_nsfw-weights.npy",batchsize=None,tag=''):
        self.tag = tag
        if weights_path == None:
            print("did not load weights")
        else:
            self.weights = np.load(weights_path, encoding="latin1").item()
        self.input_tensor = None
        self.input = tf.placeholder(tf.float32,
                                        shape=[batchsize, 448, 448, 3],
                                        name="input")
        self.input_tensor = self.input
        x = self.input_tensor
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'CONSTANT')
        x = self.__conv2d("conv_1", x, filter_depth=64,
                          kernel_size=7, stride=2, padding='valid')

        x = self.__batch_norm("bn_1", x)
        x = tf.nn.relu(x)

        x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same')

        x = self.__conv_block(stage=0, block=0, inputs=x,
                              filter_depths=[32, 32, 128],
                              kernel_size=3, stride=1)

        x = self.__identity_block(stage=0, block=1, inputs=x,
                                  filter_depths=[32, 32, 128], kernel_size=3)
        x = self.__identity_block(stage=0, block=2, inputs=x,
                                  filter_depths=[32, 32, 128], kernel_size=3)
        print(x)
        self.sup1 = self.__supervision(stage=0, size=112,resape_size=128, inputs=x)

        x = self.__conv_block(stage=1, block=0, inputs=x,
                              filter_depths=[64, 64, 256],
                              kernel_size=3, stride=2)

        x = self.__identity_block(stage=1, block=1, inputs=x,
                                  filter_depths=[64, 64, 256], kernel_size=3)
        x = self.__identity_block(stage=1, block=2, inputs=x,
                                  filter_depths=[64, 64, 256], kernel_size=3)
        x = self.__identity_block(stage=1, block=3, inputs=x,
                                  filter_depths=[64, 64, 256], kernel_size=3)
        print(x)
        self.sup2 = self.__supervision(stage=1, size=56,resape_size=256, inputs=x)

        x = self.__conv_block(stage=2, block=0, inputs=x,
                              filter_depths=[128, 128, 512],
                              kernel_size=3, stride=2)
        x = self.__identity_block(stage=2, block=1, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)
        x = self.__identity_block(stage=2, block=2, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)
        x = self.__identity_block(stage=2, block=3, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)
        x = self.__identity_block(stage=2, block=4, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)
        x = self.__identity_block(stage=2, block=5, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)
        print(x)
        self.sup3 = self.__supervision(stage=2, size=28, resape_size=512, inputs=x)

        x = self.__conv_block(stage=3, block=0, inputs=x,
                              filter_depths=[256, 256, 1024], kernel_size=3,
                              stride=2)
        x = self.__identity_block(stage=3, block=1, inputs=x,
                                  filter_depths=[256, 256, 1024],
                                  kernel_size=3)
        x = self.__identity_block(stage=3, block=2, inputs=x,
                                  filter_depths=[256, 256, 1024],
                                  kernel_size=3)
        print(x)
        x = tf.layers.average_pooling2d(x, pool_size=14, strides=1,
                                        padding="valid", name="pool")

        x = tf.reshape(x, shape=(-1, 1024))

        self.features = x
        self.logits = self.__fully_connected(name="fc_nsfw",
                                             inputs=x, num_outputs=1)
        print(self.logits)
        self.predictions = tf.nn.softmax(self.logits, name="predictions")




    """Get weights for layer with given name
    """
    def __get_weights(self, layer_name, field_name):
        if not layer_name in self.weights:
            raise ValueError("No weights for layer named '{}' found"
                             .format(layer_name))

        w = self.weights[layer_name]
        if not field_name in w:
            raise (ValueError("No entry for field '{}' in layer named '{}'"
                              .format(field_name, layer_name)))

        return w[field_name]

    """Layer creation and weight initialization
    """

    def __supervision(self, stage,size,resape_size, inputs):
        supervision_x = tf.layers.average_pooling2d(inputs, pool_size=size, strides=1,
                                        padding="valid", name="stage{}_pool".format(stage))
        print('__supervision_1',supervision_x)
        supervision_x = tf.reshape(supervision_x, shape=(-1, resape_size))
        print('__supervision_2', supervision_x)

        supervision_x = self.__fully_connected(name="stage{}_fc_nsfw".format(stage),
                                             inputs=supervision_x, num_outputs=1)
        return supervision_x


    def __fully_connected(self, name, inputs, num_outputs):
        return tf.layers.dense(
            inputs=inputs, units=num_outputs, name=name + self.tag,activation=tf.nn.sigmoid)
    def __conv2d(self, name, inputs, filter_depth, kernel_size, stride=1,
                 padding="same", trainable=False):

        if padding.lower() == 'same' and kernel_size > 1:
            if kernel_size > 1:
                oh = inputs.get_shape().as_list()[1]
                h = inputs.get_shape().as_list()[1]

                p = int(math.floor(((oh - 1) * stride + kernel_size - h)//2))

                inputs = tf.pad(inputs,
                                [[0, 0], [p, p], [p, p], [0, 0]],
                                'CONSTANT')
            else:
                raise Exception('unsupported kernel size for padding: "{}"'
                                .format(kernel_size))

        return tf.layers.conv2d(
            inputs, filter_depth,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride), padding='valid',
            activation=None, trainable=trainable, name=name + self.tag)

    def __batch_norm(self, name, inputs, training=False):
        return tf.layers.batch_normalization(
            inputs, training=training, epsilon=self.bn_epsilon,
            name=name + self.tag)

    """ResNet blocks
    """
    def __conv_block(self, stage, block, inputs, filter_depths,
                     kernel_size=3, stride=2):
        filter_depth1, filter_depth2, filter_depth3 = filter_depths

        conv_name_base = "conv_stage{}_block{}_branch".format(stage, block)
        bn_name_base = "bn_stage{}_block{}_branch".format(stage, block)
        shortcut_name_post = "_stage{}_block{}_proj_shortcut" \
                             .format(stage, block)

        shortcut = self.__conv2d(
            name="conv{}".format(shortcut_name_post), stride=stride,
            inputs=inputs, filter_depth=filter_depth3, kernel_size=1,
            padding="same"
        )

        shortcut = self.__batch_norm("bn{}".format(shortcut_name_post),
                                     shortcut)

        x = self.__conv2d(
            name="{}2a".format(conv_name_base),
            inputs=inputs, filter_depth=filter_depth1, kernel_size=1,
            stride=stride, padding="same",
        )
        x = self.__batch_norm("{}2a".format(bn_name_base), x)
        x = tf.nn.relu(x)

        x = self.__conv2d(
            name="{}2b".format(conv_name_base),
            inputs=x, filter_depth=filter_depth2, kernel_size=kernel_size,
            padding="same", stride=1
        )
        x = self.__batch_norm("{}2b".format(bn_name_base), x)
        x = tf.nn.relu(x)

        x = self.__conv2d(
            name="{}2c".format(conv_name_base),
            inputs=x, filter_depth=filter_depth3, kernel_size=1,
            padding="same", stride=1
        )
        x = self.__batch_norm("{}2c".format(bn_name_base), x)

        x = tf.add(x, shortcut)

        return tf.nn.relu(x)

    def __identity_block(self, stage, block, inputs,
                         filter_depths, kernel_size):
        filter_depth1, filter_depth2, filter_depth3 = filter_depths
        conv_name_base = "conv_stage{}_block{}_branch".format(stage, block)
        bn_name_base = "bn_stage{}_block{}_branch".format(stage, block)

        x = self.__conv2d(
            name="{}2a".format(conv_name_base),
            inputs=inputs, filter_depth=filter_depth1, kernel_size=1,
            stride=1, padding="same",
        )

        x = self.__batch_norm("{}2a".format(bn_name_base), x)
        x = tf.nn.relu(x)

        x = self.__conv2d(
            name="{}2b".format(conv_name_base),
            inputs=x, filter_depth=filter_depth2, kernel_size=kernel_size,
            padding="same", stride=1
        )
        x = self.__batch_norm("{}2b".format(bn_name_base), x)
        x = tf.nn.relu(x)

        x = self.__conv2d(
            name="{}2c".format(conv_name_base),
            inputs=x, filter_depth=filter_depth3, kernel_size=1,
            padding="same", stride=1
        )
        x = self.__batch_norm("{}2c".format(bn_name_base), x)

        x = tf.add(x, inputs)
        return tf.nn.relu(x)


def log_loss(Y, Y_pred):
    with tf.variable_scope('loss') as scope:
        Y = tf.reshape(Y, [tf.shape(Y)[0], -1])
        Y_pred = tf.reshape(Y_pred, [tf.shape(Y_pred)[0], -1])
        eps = 1e-6
        loss = -(tf.multiply(Y, tf.log(Y_pred + eps)) + tf.multiply(1 - Y, tf.log(1 - Y_pred + eps)))
    return tf.reduce_mean(loss)

def supervision_loss(Y, Y_pred,sup1,sup2,sup3):

    with tf.variable_scope('loss') as scope:
        Y = tf.reshape(Y, [tf.shape(Y)[0], -1])
        eps = 1e-6
        sup1 = tf.reshape(sup1, [tf.shape(sup1)[0], -1])
        loss_sup1 = -(tf.multiply(Y, tf.log(sup1 + eps)) + tf.multiply(1 - Y, tf.log(1 - sup1 + eps)))

        sup2 = tf.reshape(sup2, [tf.shape(sup2)[0], -1])
        loss_sup2 = -(tf.multiply(Y, tf.log(sup2 + eps)) + tf.multiply(1 - Y, tf.log(1 - sup2 + eps)))

        sup3 = tf.reshape(sup3, [tf.shape(sup3)[0], -1])
        loss_sup3 = -(tf.multiply(Y, tf.log(sup3 + eps)) + tf.multiply(1 - Y, tf.log(1 - sup3 + eps)))

        Y_pred = tf.reshape(Y_pred, [tf.shape(Y_pred)[0], -1])
        loss_last = -(tf.multiply(Y, tf.log(Y_pred + eps)) + tf.multiply(1 - Y, tf.log(1 - Y_pred + eps)))
        loss = tf.multiply(0.2,loss_sup1) + tf.multiply(0.2,loss_sup2) + tf.multiply(0.2,loss_sup3) + tf.multiply(0.4,loss_last)
    return tf.reduce_mean(loss)


def focal_loss(labels, logits, gamma, alpha, normalize = True):
    labels = tf.reshape(labels, [tf.shape(labels)[0]])

    logits = tf.reshape(logits, [tf.shape(logits)[0]])
    labels = tf.cast(labels, tf.float32)
    ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)

    alpha_t = tf.ones_like(logits) * alpha
    alpha_t = tf.where(labels > 0, alpha_t, 1.0 - alpha_t)
    probs_t = tf.where(labels > 0, logits, 1.0 - logits)####

    focal_matrix = alpha_t * tf.pow((1.0 - probs_t), gamma)
    loss = focal_matrix * ce_loss

    loss = tf.reduce_sum(loss)
    if normalize:
        n_pos = tf.reduce_sum(labels)
        total_weights = tf.stop_gradient(tf.reduce_sum(focal_matrix))
        total_weights = tf.Print(total_weights, [n_pos, total_weights])
        def has_pos():
            return loss / tf.cast(n_pos, tf.float32)
        def no_pos():
            #total_weights = tf.stop_gradient(tf.reduce_sum(focal_matrix))
            #return loss / total_weights
            return loss
        loss = tf.cond(n_pos > 0, has_pos, no_pos)
    return ce_loss

def trainning(loss, learning_rate):
    with tf.variable_scope('train_op') as scope:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(Y, Y_pred):
    Y = tf.reshape(Y, [tf.shape(Y)[0], -1])
    Y_pred = tf.reshape(Y_pred, [tf.shape(Y_pred)[0], -1])
    Y_pred = tf.round(Y_pred)
    correct_prediction = tf.equal(Y, Y_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

