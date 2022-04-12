from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
import tensorflow as tf


def model_NN(inp_sh, out_sh , activ='sigmoid', n1=64, l2=0, drop = 0):
    if (activ == 'relu'):
        init = 'he_uniform'
    else:
        init = tf.keras.initializers.glorot_uniform
    inputs = tf.keras.Input(shape=(inp_sh,))
    x = tf.keras.layers.Dense(n1,  activation=activ,kernel_initializer=init, kernel_regularizer=regularizers.l2(l2))(inputs)
    x2 = tf.keras.layers.Dense(n1,  activation=activ,kernel_initializer=init, kernel_regularizer=regularizers.l2(l2))(x)
    x2 = Dropout(drop)(x2)
    x3 = tf.keras.layers.Dense(n1,  activation=activ,kernel_initializer=init, kernel_regularizer=regularizers.l2(l2))(x2)
    x3 = Dropout(drop)(x3)
    x4 = tf.keras.layers.Dense(n1,  activation=activ,kernel_initializer=init, kernel_regularizer=regularizers.l2(l2))(x3)
    x4 = Dropout(drop)(x4)
    x5 = tf.keras.layers.Dense(n1,  activation=activ,kernel_initializer=init, kernel_regularizer=regularizers.l2(l2))(x4)
    x5 = Dropout(drop)(x5)
    outputs = tf.keras.layers.Dense(out_sh)(x5)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    opt= tf.keras.optimizers.Adam()
    model.compile(loss= tf.keras.losses.MSE, metrics = ['mse'], optimizer=opt)
    return model

