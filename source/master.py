import os, sys, re, json
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

import seaborn as sns

sys.path.append('./modules')
from fh_cnn_regressor import *


def main(df):



	if "models" in globals():
		pass
	else :

		global model_names 
		model_names = [x for x in os.listdir("./models") if x[:2]=='FH']

		global models
		models = {}
		for model_name in model_names:
			global model
			model = {}
			model['horizon'] = int(model_name.split('_')[-1])

			#load deepl_model
			deepl_model = tf.keras.models.load_model(f'./models/{model_name}/deepl_model', custom_objects={"weighted_mae": weighted_mae})
			model['deepl_model'] = deepl_model

			#load /FeatureEngineering.pkl
			with open(f'./models/{model_name}/FeatureEngineering.pkl', 'rb') as f:
				f_eng = pickle.load(f)
			model['f_eng'] = f_eng

			#load TsEngineering.pkl
			with open(f'./models/{model_name}/TsEngineering.pkl', 'rb') as f:
				ts_eng = pickle.load(f)
			model['ts_eng'] = ts_eng


			#load CustomScaler.pkl
			with open(f'./models/{model_name}/CustomScaler.pkl', 'rb') as f:
				scaler = pickle.load(f)
			model['scaler'] = scaler

			models[model_name] = model

		global ensembler
		ensembler = tf.keras.models.load_model(f'./models/ensembler_model')


	df['DateTime'] = pd.to_datetime(df['DateTime'])
	df['Forecast_date'] = pd.to_datetime(df['Forecast_date'])
	df.set_index('DateTime', inplace=True)

	X = {model_name : model['scaler'].transform(model['f_eng'].transform(df)) for model_name, model in models.items()}

	X = {model_name : format_input(X[model_name], df['Forecast_date'], df['station_code'], horizon=model['horizon']) \
		for model_name, model in models.items()}

	y = [model['deepl_model'].predict(X[model_name]) for model_name, model in models.items()]

	y = np.array([np.concatenate([_y, np.zeros((_y.shape[0], int(360-_y.shape[1])))], axis=1) for _y in y])

	_X = np.moveaxis(y, 0,2)

	prediction = ensembler.predict(_X)
	prediction = prediction[:,:,0]

	prediction = models[model_names[0]]['ts_eng'].inverse_transform(prediction)

	output = prediction.tolist()

	output_payload = {'predictions':[{'fields':[f'h+{i}' for i in range(1, 361)], 'values':output}]}

	return output_payload

