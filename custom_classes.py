# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from tensorflow.python.keras.engine import base_layer
import tensorflow as tf


class AddCoords(base_layer.Layer):
    """Add coords to a tensor"""
    def __init__(self, x_dim=30, y_dim=30, with_r=False, **kwargs):
        super(AddCoords, self).__init__(**kwargs)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def get_config(self):

        config = super(AddCoords, self).get_config().copy()
        config.update({
            'x_dim': self.x_dim,
            'y_dim': self.y_dim,
            'with_r': self.with_r,
        })
        return config
     

    def call(self, input_tensor):
        """
        input_tensor: (batch, x_dim, y_dim, c)
        """
        batch_size_tensor = tf.shape(input_tensor)[0]
        xx_ones = tf.ones([batch_size_tensor, self.x_dim], dtype=tf.int32)
        xx_ones = tf.expand_dims(xx_ones, -1)
 
        xx_range = tf.tile(tf.expand_dims(tf.range(self.x_dim), 0), [batch_size_tensor, 1])
        xx_range = tf.expand_dims(xx_range, 1)
 
        xx_channel = tf.matmul(xx_ones, xx_range)
        xx_channel = tf.expand_dims(xx_channel, -1)
 
        yy_ones = tf.ones([batch_size_tensor, self.y_dim], dtype=tf.int32)
        yy_ones = tf.expand_dims(yy_ones, 1)
 
        yy_range = tf.tile(tf.expand_dims(tf.range(self.y_dim), 0), [batch_size_tensor, 1])
        yy_range = tf.expand_dims(yy_range, -1)
 
        yy_channel = tf.matmul(yy_range, yy_ones)
        yy_channel = tf.expand_dims(yy_channel, -1)
        xx_channel = tf.cast(xx_channel, 'float32') / (self.x_dim - 1)
        yy_channel = tf.cast(yy_channel, 'float32') / (self.y_dim - 1)
        xx_channel = xx_channel*2 - 1
        yy_channel = yy_channel*2 - 1
 
        ret = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)
 
        if self.with_r:
            rr = tf.sqrt( tf.square(xx_channel-0.5) + tf.square(yy_channel-0.5))
            ret = tf.concat([ret, rr], axis=-1)
      
        return ret
 
 
class CoordConv(base_layer.Layer):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim=15, y_dim=15, with_r=False, *args, **kwargs):
        super(CoordConv, self).__init__()
        for i, v in kwargs.items():
            if i == 'addcoords':
                self.addcoords = v
            if i == 'conv':
                self.conv = v

    def get_config(self):
        config = super(CoordConv, self).get_config().copy()
        config.update({
            'addcoords': self.addcoords,
            'conv': self.conv,

        })
        return config
    
    def call(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret


class CustomScaler(BaseEstimator,TransformerMixin): 
    # note: returns the feature matrix with the binary columns ordered first  
    def __init__(self,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.dummies = 0


    def fit(self, X, dum_num, y=None):
        self.dummies = dum_num
        self.scaler.fit(X.iloc[:,:self.dummies], y)
        return self

    def transform(self, X, y=None, copy=None):
        X_tail = self.scaler.transform(X.iloc[:,:self.dummies])
        return np.concatenate((X.iloc[:,self.dummies:],X_tail), axis=1)
