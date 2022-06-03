import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import r2_score
from  sklearn.linear_model import LinearRegression, LogisticRegression


def prediction_lin_reg(df):
    x = df.iloc[:, 5].values.reshape(-1,1) #years
    y = df.iloc[:, 9].values.reshape(-1,1) #total population
    model = LinearRegression().fit(x,y)
    return model

def prediction_log_reg(df):
    x = df.iloc[:, 5].values.reshape(-1,1) #years
    y = df.iloc[:, 9].values.reshape(-1,1) #total population
    model = LogisticRegression().fit(x,y)
    return model

def dataframe_gen(df, area_list):
    frames = {}
    model_dict_lin = {}
    model_dict_log = {}
    for area in area_list:
        frames[area] = df.loc[df['Location'] == area]
    for area in area_list:
        model_dict_lin[area] = prediction_lin_reg(frames[area])
       # model_dict_log[area] = prediction_log_reg(frames[area])

    return model_dict_lin#, model_dict_log


def prediction(model_dict, years_to_predict):
    oficial_dict = {}
    for key in model_dict.keys():
        country_dict ={}
        for year in range(len(years_to_predict)):
            country_dict[years_to_predict[year]] = int(model_dict[key].coef_[0][0]* years_to_predict[year] + model_dict[key].intercept_[0])
        oficial_dict[key] = country_dict
    return oficial_dict


#return years_to_predict

def main():
    #read the dataset
    #df = pd.read_csv('../raw_data/df_test_2.csv')
    df = pd.read_csv('raw_data/df_test_2.csv')
    #select the area into a list

    #def years():
    year_start = 2021
    year_end = 2050
    num = year_end-year_start
    years_to_predict = []

    for i in range (num+1):
        amount = year_start + i
        years_to_predict.append(amount)
        i += 1
    area_list = df.Location.unique().tolist()
    #segment the dataFrame per country or region
    #dic_list_lin,dic_list_log=dataframe_gen(df,area_list)
    dic_list_lin=dataframe_gen(df,area_list)

    #model = prediction_model(df)
    #dic_list_lin=dataframe_gen(df, area_list) ##this is the one that we are not sure

    #model = prediction_lin_reg(model_dict)
    result_lin = prediction(dic_list_lin,years_to_predict)
    #result_log = prediction(dic_list_log,years_to_predict)

    result_lin=pd.DataFrame(result_lin)
    result_lin.T
    result_lin.T.to_csv('raw_data/results_lin.csv')

    # result_log=pd.DataFrame(result_log)
    # result_log.T
    # result_log.T.to_csv('raw_data/results_log.csv')


if __name__ == "__main__":
    main()
