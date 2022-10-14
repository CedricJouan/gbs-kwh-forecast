
import os, sys, re, json, random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin




def set_seed(seed_value = 42):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)  


class FeatureEngineering(BaseEstimator, TransformerMixin):

    def __init__(self):
        
        self.phrase_encoder = None
        self.day_night_encoder = {'D':1, 'N':0}
        self.columns = ['zenith', 'azimuth', 'Fcst Hour', 'Day or Night',
                        'Temp', 'DewPt', '% chance of Precipitation', 'Precipitation Amount',
                        'Relative Humidity (%)', 'Wind Speed', 'Wind Direction',
                        'Cloud Cover', 'Visibility', 'Mean Sea Level Pressure', 'Weather Phrase(Short)']
        
    def fit(self, X, y = None):
        self.phrase_encoder = X.groupby('Weather Phrase(Short)').apply(lambda t : t['Cloud Cover'].mean()).to_dict()
        return self

    def transform(self, X, y = None):
        x = X[self.columns].copy()
        
        x['Day or Night'] = x['Day or Night'].apply(lambda t : self.day_night_encoder[t])
        x['Weather Phrase(Short)'] = x['Weather Phrase(Short)'].apply(lambda t : self.phrase_encoder[t])
        
        x = x.astype(float)
        
        x['wx'] = x.apply(lambda z : (z['Wind Speed'])*(np.cos(z['Wind Direction']*(np.pi/180))), axis=1)
        x['wy'] = x.apply(lambda z : (z['Wind Speed'])*(np.sin(z['Wind Direction']*(np.pi/180))), axis=1)
        x.drop(['Wind Direction', 'Wind Speed'], axis=1, inplace=True)
        
        x['sqrtZenith'] = np.sqrt(x['zenith'])

        return x

    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)

class TsEngineering():
    def __init__(self, sqrt_transform=False):

        self.sqrt_transform = sqrt_transform

    def fit(self, Y):
        return self

    def transform(self, Y):
        y = Y[['kWh']].copy()     
        if self.sqrt_transform :
            y = np.sqrt(1+y/1000.)

        y = y/5000.

        return y

    def fit_transform(self, Y):
        y = Y.copy()
        _ = self.fit(y)
        return self.transform(y)

    def inverse_transform(self, Y):
        y = Y.copy()

        y=y*5000.

        if self.sqrt_transform :
            y = (y**2 -1)*1000.

        return y


class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.avg = None
        self.std = None 
        self.features = None

    def fit(self, X, y = None):
        self.avg = X.mean(axis=0)
        self.std = X.std(axis=0)
        
        self.features = list(X.columns)
        return self

    def transform(self, X, y = None):
        return (X-self.avg)/self.std

    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)



def format_input(data, forecast_date, station_id, horizon=None):
    x = data.copy()
    x['Forecast_date'] =  forecast_date 
    x['station_code'] = station_id   
    
    output = np.stack(x.groupby('station_code').apply(lambda x : np.stack(x.drop('station_code', axis=1)\
                        .groupby('Forecast_date').apply(lambda x : x.drop('Forecast_date', axis=1).values).values)).values, axis=-1)
    
    if output.shape[-2]==1:
        #y
        output = output[:,:,0,0]
        if horizon:
            if horizon == 360:
                output = output[:,-360:]
            else :
                output = output[:,-360:-(360 - horizon)]
    else :
        #X
        output = np.moveaxis(output, 1, 2)
        if horizon:
            if horizon != 360:
                output = output[:,:,:-(360 - horizon),:]                

            output = np.moveaxis(output[:,:,:,0],1 ,2)
            
    return output


class SeqSeqSingleSationsCNN(tf.keras.Model):
    def __init__(self,*, cnn_reg_filter, dnn_reg_units, conv_reg_width, features,\
                        dense_dropout_rate=0.3, cnn_dropout_rate=0.1 ,horizon=360, batch_norm=False):
        super(SeqSeqSingleSationsCNN, self).__init__()

        self.conv_reg_width = conv_reg_width
        self.cnn_reg_filter = cnn_reg_filter
        self.dnn_reg_units = dnn_reg_units
        self.regressor_input_shape=(horizon, len(features))

        self.cnn = tf.keras.layers.Conv1D(filters=cnn_reg_filter, 
                                                  kernel_size=(conv_reg_width,), 
                                                  input_shape = self.regressor_input_shape, 
                                                  activation='relu')
        self.batch_norm = batch_norm
        self.batch_norm_layer = tf.keras.layers.BatchNormalization()

        self.hidden_dense = tf.keras.layers.Dense(units=dnn_reg_units, activation='relu')
        self.out_dense = tf.keras.layers.Dense(1)

        self.conv_dropout = tf.keras.layers.Dropout(cnn_dropout_rate)
        self.dense_dropout = tf.keras.layers.Dropout(dense_dropout_rate)

        self.reshape_output = tf.keras.layers.Reshape(target_shape=(horizon,))

    def call(self, x, training=False):
        x = x[:,23-(self.conv_reg_width-1):,:]
        x = self.cnn(x)
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        if training:
            x = self.conv_dropout(x)
        x = self.hidden_dense(x)
        if training:
            x = self.dense_dropout(x)
        x = self.out_dense(x)
        return self.reshape_output(x)


def weighted_mae(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype='float32')
    y_pred = tf.convert_to_tensor(y_pred, dtype='float32')

    ae = tf.abs(y_true - y_pred)
    mae = tf.reduce_mean(ae, axis=0)

    horizon_weights = [(HORIZON - h) for h in range(HORIZON)]
    horizon_weights = tf.constant(horizon_weights, dtype='float32')
    horizon_weights = horizon_weights/tf.reduce_sum(horizon_weights)

    return tf.tensordot(mae, horizon_weights, axes = [0, 0])


def train_model(x, y, model, max_epochs=50, validation_split=0.3, batch_size=64, loss="mae", verbose=2):
    HORIZON = y.shape[-1]

    def weighted_mae(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype='float32')
        y_pred = tf.convert_to_tensor(y_pred, dtype='float32')
    
        ae = tf.abs(y_true - y_pred)
        mae = tf.reduce_mean(ae, axis=0)
    
        horizon_weights = [(HORIZON- h) for h in range(HORIZON)]
        horizon_weights = tf.constant(horizon_weights, dtype='float32')
        horizon_weights = horizon_weights/tf.reduce_sum(horizon_weights)

        return tf.tensordot(mae, horizon_weights, axes = [0, 0])

    if loss == "weighted_mae":
        loss = weighted_mae

    model.compile(optimizer="Adam", loss=loss)
    history = model.fit(x=x, y=y, 
                        validation_split=validation_split, 
                        batch_size=batch_size, 
                        epochs=max_epochs,
                        verbose=verbose)
    return history



