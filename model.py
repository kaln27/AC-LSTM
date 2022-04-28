import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class Attention(layers.Layer):
    def __init__(self, X_shape) -> None:
        #input_shape: [batch_size, time_step, feature] => [batch_size, feature]
        super().__init__()
        self.time_step, self.hidden_size = X_shape[1], X_shape[2]
        self.h_t = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_size, 1], stddev=0.5, dtype=tf.float32))
        # H_t: [feature, 1]
        self.W = tf.Variable(tf.random.truncated_normal(shape=[self.hidden_size, self.hidden_size], stddev=0.5, dtype=tf.float32))
        # W: [feature, feature]
    def call(self, X):
        score = tf.matmul(tf.matmul(tf.reshape(X,[-1,self.hidden_size]), self.W), self.h_t)
        score = tf.reshape(score,[-1,self.time_step,1])
        alpha = tf.nn.softmax(score, axis=1)
        c_t = tf.matmul(tf.transpose(X, [0, 2, 1]), alpha) # [batch_size, feature, 1]
        c_t = tf.squeeze(c_t, axis=2) # [batch_size, feature]
        return c_t



class Model(keras.Model):
    def __init__(self, input_shape):
        super(Model, self).__init__()
        self.time_step, self.hidden_size = input_shape[1], input_shape[2]
        self.init1 = keras.initializers.GlorotNormal()
        self.init2 = keras.initializers.GlorotNormal()
        self.init3 = keras.initializers.GlorotNormal()
        self.lstm1 = layers.LSTM(units=64, return_sequences=True, activation=tf.nn.relu, 
        kernel_initializer='glorot_uniform', recurrent_initializer=self.init1) # (b, t, feature) => (b, t, 64)
        self.lstm2 = layers.LSTM(units=64, return_sequences=True, activation=tf.nn.relu,
        kernel_initializer='glorot_uniform', recurrent_initializer=self.init2) # (b, t, 64) => (b, t, 64)
        self.lstm3 = layers.LSTM(units=64, return_sequences=True, activation=tf.nn.relu,
        kernel_initializer='glorot_uniform', recurrent_initializer=self.init3) # (b, t, 64) => (b, t, 64)
        self.att1 = Attention(X_shape=[None, self.time_step, 64])
        self.att2 = Attention(X_shape=[None, self.time_step, 64])
        self.att3 = Attention(X_shape=[None, self.time_step, 64])
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.dense = layers.Dense(units=1, kernel_initializer='glorot_normal')

    def call(self, inputs):
        x = self.lstm1(inputs) # (b, t, 64)
        x = self.bn1(x)
        att1 = self.att1(x) # (b, 64)
        x = self.lstm2(x) # (b, t, 64)
        x = self.bn2(x)
        att2 = self.att2(x) # (b, 64)
        x = self.lstm3(x) # (b, t, 64)
        x = self.bn3(x)
        att3 = self.att3(x) # (b, 64)
        x = tf.concat([att1, att2, att3], axis=1) # (b, 192)
        x = self.dense(x) # (b, 1)
        return x



if __name__ == '__main__':
    pass
