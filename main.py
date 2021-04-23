import pandas as pd
import numpy as np
from model import Model
import tensorflow as tf
import tensorflow.keras as keras
import math

checkpoint_path = "./saved_model/save"
load = False
def get_data(training_data_size):
    data = pd.read_csv("./data/train.csv")
    data = data.to_numpy()
    label = np.zeros((data.shape[0],10))
    index = np.arange(0,data.shape[0],1)
    label[index,data[:,0][index]]=1

    image = data[:,1:].copy()
    image = image.reshape((data.shape[0],28,28))
    
    training_image, training_label = image[:int(data.shape[0]*training_data_size)], label[:int(data.shape[0]*training_data_size)]
    testing_image, testing_label = image[int(data.shape[0]*training_data_size):],label[int(data.shape[0]*training_data_size):]

    training_image=np.expand_dims(training_image,axis=-1)
    testing_image=np.expand_dims(testing_image,axis=-1)
    return training_image.astype(np.float), training_label, testing_image.astype(np.float), testing_label

def train(load):
    lr = 0.05
    model = Model()

    log_path = "./log/"
    if load:
        try:
            model.load_weights(checkpoint_path)
            print("Load successful, start training")
        except Exception as e:
            print(e)
            print("Start with new model")
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,save_weights_only=True,verbose=1,save_best_only=True)

    training_image, training_label, testing_image, testing_label = get_data(0.75)
    model.fit(x=training_image,y=training_label,batch_size=128,epochs=100,validation_data=(testing_image,testing_label),callbacks=[checkpoint_callback,tensorboard_callback])

if __name__=="__main__":
    train(load)