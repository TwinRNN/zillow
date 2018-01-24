
from keras.layers import Dense,LSTM
from keras.models import Sequential
from keras.utils import multi_gpu_model
import dataset_zillow
import numpy as np 
import pandas as pd 
import tensorflow as tf 

batch_size = 128
seqlen =20
units =256


def build_model():
	with tf.device('/cpu:0'):
		model=Sequential()
		model.add(LSTM(units,return_sequences=True,input_shape=[time_steps,feature_size]))
		model.add(LSTM(units,return_sequences=True,input_shape=[time_steps,feature_size]))
		model.add(Dense(1))
	return model

def parallel_model(model):
	parallel_model = multi_gpu_model(model,gpus=2)
	return parallel_model



def train():
	batch_size = 128
	seqlen =20
	units =256

	dataset = dataset_zillow.Dataset(batch_size,seqlen)
	dataset.split(3600,117600,3600,3600,0)
	total_batch = int(120000/batch_size)
	X_train=list()
	Y_train=list()
	for batchidx in range(total_batch):
		
		X_train_idx,Y_train_idx = dataset.next_batch()
		X_train_idx=X_train_idx.tolist()
		Y_train_idx = Y_train_idx.tolist()
		X_train.append(X_train_idx)
		Y_train.append(Y_train_idx)
		X_train = np.stack(X_train,axis=0)
		Y_train = np.stack(Y_train,axis=0)
	print(X_train.shape)
	model = build_model()
	parallel_model = parallel_model(model)
	parallel_model.compile(loss="mse",optimizer="SGD")
	parallel_model.fit(X_train,Y_train,epochs=20,batch_size=128)

train()
