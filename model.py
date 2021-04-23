import tensorflow as tf
import tensorflow.keras as keras

class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        initializer = tf.keras.initializers.GlorotUniform()
        self.net = keras.Sequential([keras.layers.Conv2D(128,3,padding="same",kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    keras.layers.Conv2D(64,3,padding="same",kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    keras.layers.Conv2D(64,3,padding="same",kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    tf.keras.layers.Dropout(0.3),

                                    keras.layers.Conv2D(64,4,kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    tf.keras.layers.Dropout(0.3),

                                    keras.layers.Conv2D(128,3,padding="same",kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    keras.layers.Conv2D(128,4,padding="same",kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    keras.layers.Conv2D(128,3,padding="same",kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    tf.keras.layers.Dropout(0.3),

                                    keras.layers.Conv2D(128,5,kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    tf.keras.layers.Dropout(0.3),

                                    keras.layers.Conv2D(256,3,padding="same",kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    keras.layers.Conv2D(256,4,padding="same",kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    keras.layers.Conv2D(256,3,padding="same",kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    tf.keras.layers.Dropout(0.3),

                                    keras.layers.Conv2D(256,4,kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    tf.keras.layers.Dropout(0.3),

                                    keras.layers.Flatten(),
                                    keras.layers.Dense(256,kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    tf.keras.layers.Dropout(0.1),
                                    keras.layers.Dense(256,kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    tf.keras.layers.Dropout(0.1),
                                    keras.layers.Dense(128,kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU(),
                                    tf.keras.layers.Dropout(0.1),
                                    keras.layers.Dense(10,kernel_initializer=initializer),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.Softmax()
        ])
    def call(self, inputs,training=True):
        return self.net(inputs,training=training)