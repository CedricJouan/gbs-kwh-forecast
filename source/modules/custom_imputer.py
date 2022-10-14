
import pandas as pd
import math
import numpy as np
import os

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


from astral import LocationInfo
from astral.location import Location


def custom_imputer(solar_data, co_weather, site='Martin', train_test_split_date=None, optimize=False):

    if 'DateTime' in solar_data.columns:
        df_solar = solar_data[['DateTime', 'kWh']].copy()
    else :
        df_solar = solar_data.copy()
        df_solar['DateTime']=df_solar.index
        df_solar.reset_index(drop=True, inplace =True)
        df_solar = df_solar[['DateTime', 'kWh']].copy()


    if site == 'Martin':
        outliers = [('2018-01-23 16:00:00', '2018-01-24 13:00:00'),
                    ('2018-08-13 19:00:00', '2018-08-14 17:00:00'),
                    ('2019-01-09 09:00:00', '2019-01-10 08:00:00'),
                    ('2019-04-26 18:00:00', '2019-04-27 17:00:00'),
                    ('2020-05-22 11:00:00', '2020-06-03 10:00:00'),
                    ('2019-07-03 19:00:00', '2019-07-18 08:00:00'),
                    ('2019-08-14 01:00:00', '2019-08-29 00:00:00')]

    else :
        print('Not implemented yet - Returning input data')
        return df_solar.kWh.values

    abnormal_long_zero_segments = [(pd.to_datetime(x), pd.to_datetime(y)) for (x,y) in outliers]

    outlier_mask = np.array([len(df_solar[(df_solar['DateTime']>=x)&(df_solar['DateTime']<=y)])>0 for (x,y) in outliers])
    outliers = list(np.array(outliers)[outlier_mask])

    if len(outliers) == 0:
        print('No outliers for this data frame')
        return df_solar.kWh.values

    location_info = LocationInfo("Martin", "Martin", "US/Eastern", 37.876, -78.061)
    location = Location(location_info)
    df_solar["zenith"] = df_solar.apply(lambda x : location.solar_zenith(x['DateTime']), axis=1)
    df_solar["azimuth"] = df_solar.apply(lambda x : location.solar_azimuth(x['DateTime']), axis=1)

    # loading CO data
    df_co_weather = co_weather.copy()
    df_co_weather['DateTime'] = pd.to_datetime(df_co_weather['DateTime'])
    # print(f"Shape of the CO weather data : {df_co_weather.shape}")


    weather = df_co_weather[['DateTime',
                            'CloudCoveragePercent',
                            'MslPressureMillibars',
                            'RelativeHumidityPercent',
                            'PrecipitationPreviousHourInches',
                            'SurfaceTemperatureFahrenheit',
                            'WindDirectionDegrees',
                            'WindSpeedMph']].copy()


    imputation_start_date = max(min(weather.DateTime), min(df_solar.DateTime))
    if train_test_split_date :
        imputation_end_date = min(max(weather.DateTime), max(df_solar.DateTime), train_test_split_date)
    else :
        imputation_end_date = min(max(weather.DateTime), max(df_solar.DateTime))

    weather = weather[(weather['DateTime']<imputation_end_date)&(weather['DateTime']>imputation_start_date)].copy()

    df_solar.sort_values(by='DateTime', axis=0, ascending=True, inplace =True)
    start_signal = list(df_solar[df_solar['DateTime']<=imputation_start_date].kWh.values)
    end_signal = list(df_solar[df_solar['DateTime']>=imputation_end_date].kWh.values)

    df_solar = df_solar[(df_solar['DateTime']<imputation_end_date)&(df_solar['DateTime']>imputation_start_date)].copy()

    data = df_solar.merge(weather, on='DateTime', how='left').set_index('DateTime')

    data['wx'] = data.apply(lambda x : (x['WindSpeedMph'])*(np.cos(x['WindDirectionDegrees']*(np.pi/180.))), axis=1)
    data['wy'] = data.apply(lambda x : (x['WindSpeedMph'])*(np.sin(x['WindDirectionDegrees']*(np.pi/180.))), axis=1)
    data.drop(['WindDirectionDegrees','WindSpeedMph'], axis=1, inplace=True)

    data['sqrtZenith'] = np.sqrt(data.zenith)
    data.dropna(inplace=True)

    signal = data.kWh.copy()
    features = data.drop('kWh', axis = 1).copy()



    for abnormal_long_zero_segment in abnormal_long_zero_segments :
        signal[abnormal_long_zero_segment[0]:abnormal_long_zero_segment[1]] = np.nan


    X = features[~signal.isna()].values
    y = signal.dropna().values

    #Random forest is a placeholder, the To-Date model needs to be improved!!
    if optimize :
        print('Optimizing Imputer Model')
        parameters = {'n_estimators':[100, 150, 180, 200],'ccp_alpha':[0.4, 0.5, 0.7, 0.9]}
        base_estimator = RandomForestRegressor(criterion='squared_error')
        regressor = GridSearchCV(base_estimator, parameters, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1, verbose=4)
        regressor.fit(X, y)
        print(regressor.best_params_)
        print(base_estimator)
        print(regressor)

    else :
        regressor = RandomForestRegressor(n_estimators = 180, 
                                         ccp_alpha = 0.7, 
                                         criterion='squared_error')
        regressor.fit(X, y)


    # looping is not necessary here, TO DO : set up an apply function
    imputed_kWh = []
    for i in range (len(signal)):
        if math.isnan(signal.iloc[i]):
            x = features.iloc[i:i+1].values
            impute_value = regressor.predict(x)[0]
            if impute_value < 0.1 : # should imporve the ml model instead of this direty little prediction fix
                impute_value = 0
            imputed_kWh.append(impute_value)
        else :
            imputed_kWh.append(signal.iloc[i])

    return start_signal  + imputed_kWh + end_signal