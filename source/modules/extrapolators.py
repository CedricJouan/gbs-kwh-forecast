
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
import os, sys, json, re
import pandas as pd
import numpy as np




class FourrierMultiplicativeSeasonalityEstimator(BaseEstimator):
    def __init__( self, Dy=3, Dd=6):
        self.Dy = Dy 
        self.Dd = Dd
        self.wy = ((2*np.pi)/(24.*365.))
        self.wd = ((2*np.pi)/(24.))

    def fit(self, X, y ):

        # Add a unit test on the type of the input X -> should be dataframe with DatetimeIndex and a Columns called "isDay"
        self.startTrainDate = X.index.min()
        endTrainDate = X.index.max()
        len_train_data = X.shape[0]


        dateRange = pd.date_range(self.startTrainDate, endTrainDate, freq="H")
        assert len(dateRange) == len_train_data, 'The size of the DateiIndex of the input should have the size of pd.date_range(start_date, end_date, freq="H")'


        cc = np.array([[X.isDay.iloc[t-1]*np.cos(k*self.wy*t)*np.cos(p*self.wd*t) for p in range(self.Dd) for k in range(self.Dy)] for t in range(1, len_train_data+1)])
        cs = np.array([[X.isDay.iloc[t-1]*np.cos(k*self.wy*t)*np.sin(p*self.wd*t) for p in range(self.Dd) for k in range(self.Dy)] for t in range(1, len_train_data+1)])
        ss = np.array([[X.isDay.iloc[t-1]*np.sin(k*self.wy*t)*np.sin(p*self.wd*t) for p in range(self.Dd) for k in range(self.Dy)] for t in range(1, len_train_data+1)])
        sc = np.array([[X.isDay.iloc[t-1]*np.sin(k*self.wy*t)*np.cos(p*self.wd*t) for p in range(self.Dd) for k in range(self.Dy)] for t in range(1, len_train_data+1)])
        
        X_seasonal = np.concatenate([cc, cs, ss, sc], axis=1)

        self.lr_season = LinearRegression(fit_intercept=False)
        self.lr_season.fit(X_seasonal, y)

        return self 

    def predict(self, X):

        startDate = X.index.min()
        endDate = X.index.max()
        len_pred_data = X.shape[0]

        dateRange = pd.date_range(startDate, endDate, freq="H")
        assert len(dateRange) == len_pred_data, 'The size of the DateiIndex of the input should have the size of {pd.date_range(start_date, end_date, freq="H")}'
        assert startDate >= self.startTrainDate, 'In the current implementation the prediction set should start after the training set'

        len_pred_data = len(dateRange)
        start_pred_t = int((startDate - self.startTrainDate).days*24 + (startDate - self.startTrainDate).seconds/(60*60))

        cc = np.array([[X.isDay.iloc[t-start_pred_t-1]*np.cos(k*self.wy*t)*np.cos(p*self.wd*t) for p in range(self.Dd) for k in range(self.Dy)] for t in range(start_pred_t+1, start_pred_t+len_pred_data+1)])
        cs = np.array([[X.isDay.iloc[t-start_pred_t-1]*np.cos(k*self.wy*t)*np.sin(p*self.wd*t) for p in range(self.Dd) for k in range(self.Dy)] for t in range(start_pred_t+1, start_pred_t+len_pred_data+1)])
        ss = np.array([[X.isDay.iloc[t-start_pred_t-1]*np.sin(k*self.wy*t)*np.sin(p*self.wd*t) for p in range(self.Dd) for k in range(self.Dy)] for t in range(start_pred_t+1, start_pred_t+len_pred_data+1)])
        sc = np.array([[X.isDay.iloc[t-start_pred_t-1]*np.sin(k*self.wy*t)*np.cos(p*self.wd*t) for p in range(self.Dd) for k in range(self.Dy)] for t in range(start_pred_t+1, start_pred_t+len_pred_data+1)])

        X_seasonal = np.concatenate([cc, cs, ss, sc], axis=1)

        return  self.lr_season.predict(X_seasonal)

    def predict_date(self, date, isDay):

        startDate = date
        endDate = date
        len_pred_data = 1

        dateRange = pd.date_range(startDate, endDate, freq="H")
        assert len(dateRange) == len_pred_data, 'The size of the DateiIndex of the input should have the size of {pd.date_range(start_date, end_date, freq="H")}'
        assert startDate >= self.startTrainDate, 'In the current implementation the prediction set should start after the training set'

        len_pred_data = len(dateRange)
        start_pred_t = int((startDate - self.startTrainDate).days*24 + (startDate - self.startTrainDate).seconds/(60*60))

        cc = np.array([[isDay*np.cos(k*self.wy*t)*np.cos(p*self.wd*t) for p in range(self.Dd) for k in range(self.Dy)] for t in range(start_pred_t+1, start_pred_t+len_pred_data+1)])
        cs = np.array([[isDay*np.cos(k*self.wy*t)*np.sin(p*self.wd*t) for p in range(self.Dd) for k in range(self.Dy)] for t in range(start_pred_t+1, start_pred_t+len_pred_data+1)])
        ss = np.array([[isDay*np.sin(k*self.wy*t)*np.sin(p*self.wd*t) for p in range(self.Dd) for k in range(self.Dy)] for t in range(start_pred_t+1, start_pred_t+len_pred_data+1)])
        sc = np.array([[isDay*np.sin(k*self.wy*t)*np.cos(p*self.wd*t) for p in range(self.Dd) for k in range(self.Dy)] for t in range(start_pred_t+1, start_pred_t+len_pred_data+1)])

        X_seasonal = np.concatenate([cc, cs, ss, sc], axis=1)

        output = self.lr_season.predict(X_seasonal)

        return  output[0]

