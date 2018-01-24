from keras.callbacks import Callback
from keras.layers import Dense,LSTMCell,RNN,StackedRNNCells,Input,LSTM,concatenate
from keras.layers.core import Reshape
from keras.models import Sequential,Model
from keras.utils import multi_gpu_model
import dataset_zillow as dataset
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from keras import backend as K
import gatherdata
batch_size = 128
seqlen =20
units =512


def build_model(timeseries_feature_size):
	with tf.device('/cpu:0'):
		'''
		model=Sequential()
		#model.add(LSTM(units,return_sequences=True,input_shape=[None,33]))
		#model.add(LSTM(units,return_sequences=False))
		cells = [LSTMCell(units) for _ in range(2)]
		cell = StackedRNNCells(cells)
		model.add(RNN(cell,return_sequences=True,input_shape=[None,33]))
		model.add(Dense(1))
		'''
		#build house model
		house_input = Input(shape=(None,None,33),dtype='float32')
		house_middle = LSTM(units,return_sequences = True)(house_input)
		house_output = LSTM(units,return_sequences = True)(house_middle)

		#build time model
		macro_input = Input(shape = (None,None,timeseries_feature_size),dtype='float32')
		macro_middle = LSTM(units,return_sequences = True)(macro_input)
		macro_output = LSTM(units,return_sequences = False)(macro_middle)
		macro_output = Reshape([128,20,units])(macro_output)

		#concatenate
		middle_output = concatenate([house_output,macro_output])
		finale_output = Dense(1,activation='sigmoid')(middle_output)
		model = Model(inputs=[house_input,macro_input],outputs=finale_output)
	return model

def parallel_model(model):
	parallel_model = multi_gpu_model(model,gpus=2)
	return parallel_model


def cus_loss(y_true,y_pred):
	price_error = K.abs(1 - K.exp(y_true-y_pred))
	loss = K.mean(price_error)
	return loss
def train():
	batch_size = 128
	seqlen =20
	units =256
	'''
	with open("data/zillow-model-data-original",'rb') as f:
		raw = f.read()
		data = pickle.loads(raw)
		X_train = data['train_df'][: 100000].values
		Y_train = data['logerror_df'][: 100000].values
		X_test = data['train_df'][100000:150000].values
		Y_test = data['logerror_df'][100000:150000].values
	'''
	X_train,Y_train,macro_train,X_valid,Y_valid,macro_valid,X_test,Y_test,macro_test=gatherdata.gather()
	timeseries_feature_size = macro_train.shape[-1]
	callback = [Callback.earlystopping(monitor=cus_loss,patience=5,mode='min')]
	X_train = np.reshape(X_train,[-1,20,33])
	Y_train = np.reshape(Y_train,[-1,20,1])
	macro_train = np.reshape(macro_train,[-1,20,timeseries_feature_size])
	macro_valid = np.reshape(macro_valid,[-1,20,timeseries_feature_size])
	macro_test = np.reshape(macro_test,[-1,20,timeseries_feature_size])
	print(X_train.shape)
	print(Y_train.shape)
	model = build_model(timeseries_feature_size)
	model2 = parallel_model(model)
	model2.compile(loss=cus_loss,optimizer="SGD")
	model2.fit([X_train,macro_train],Y_train,validation_data=([X_valid,macro_valid],Y_valid),batch_size=256,epoch=20,callbacks=callback)
	X_test= X_test.reshape(1,-1,33)
	print(Y_test.shape)
	Y_test = Y_test.reshape((1,-1,1))
	print(Y_test.shape)
	y=model2.predict([X_teat,macro_test])
	print(y.shape)
	acc = np.mean(np.abs(1-np.exp(Y_test-y)))
	print(acc)


train()
