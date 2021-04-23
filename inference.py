import pandas as pd
import numpy as np
from model import Model
import tensorflow as tf
import tensorflow.keras as keras
import math
checkpoint_path = "./saved_model/save"
def inference():
    data = pd.read_csv("./data/test.csv")
    data = data.to_numpy()
    data = data.reshape((data.shape[0],28,28,1))
    result = np.arange(1,data.shape[0]+1,1)
    result = np.concatenate([np.expand_dims(result,axis=1),np.zeros((data.shape[0],1))],axis = 1)
    model = Model()
    try:
        model.load_weights(checkpoint_path).expect_partial()
        print("Load successful")
    except Exception as e:
        print(e)
    
    for i in range(0,data.shape[0],128):
        result[i:i+128,1] = np.argmax(model(data[i:i+128].astype(np.float),training=False),axis=1)
    
    result[math.floor(data.shape[0]/128)*128:data.shape[0],1] = np.argmax(model(data[math.floor(data.shape[0]/128)*128:data.shape[0]].astype(np.float),training=False),axis=1)
    print(result[0:10])
    pd.DataFrame(result.astype(np.int)).to_csv("./data/test_result.csv",header=["ImageId","Label"],index=None)

if __name__=="__main__":
    inference()