import json

from pymongo import MongoClient
import pandas as pd
import numpy as np
import numpy_indexed as npi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import BaggingRegressor
from src.plot.plotter import Plotter

models = {"Linear Regression": LinearRegression, "Random Forest Regressor": RandomForestRegressor,
          "SGD Regressor": SGDRegressor, "Bayesian Ridge": BayesianRidge,
          "Bagging Regressor": BaggingRegressor, "Kernel Ridge": KernelRidge, "Decision Tree": DecisionTreeRegressor}

dataHeaders = ['applicationName', 'process', 'inputBytes', 'inputRecords', 'executors', 'shuffleWrite', 'time']

def loadData():
    client = MongoClient('localhost', 27017)
    print(str(client))
    db = client['dioneJson']
    collection = db['SparkExecutors']

    data = {'inputBytes': [], 'inputRecords': [], 'executors': [], 'numOfTasks': [], 'time': []}

    df = pd.DataFrame(columns=dataHeaders)

    for doc in collection.find():
        executors = doc['executors']
        stages = doc['stages']
        attempts = doc['attempts']
        name = doc['application_name']

        for stage in stages:
            stageID = stage['stageID']
            time = stage['time']
            inputBytes = stage['inputBytes']
            outputBytes = stage['outputBytes']
            inputRecords = stage['inputRecords']
            outputRecords = stage['outputRecords']
            numOfTasks = stage['numOfTasks']

            data['inputBytes'].append(inputBytes)
            data['inputRecords'].append(inputRecords)
            data['executors'].append(len(executors))
            data['numOfTasks'].append(numOfTasks)
            data['time'].append(time)

        df = pd.DataFrame.from_dict(data)

        # for executor in executors:
        #     execID = executor['id']
        #     inputSize = executor['input']
        #     shuffleWrite = executor['shuffleWrite']

    return df


def predict(df):
    for col in df:
        df[col] = norm(df[col])
    y = df['time']
    x = df.drop('time', 1)
    config_dictionary = json.load(open('/home/thanasis/PycharmProjects/dionePredict/analysis.json'))
    plotter = Plotter()
    test_sizes = config_dictionary['train_size']
    plotter.setup_plot(**config_dictionary)

    scores_per_technique = {}

    for model in models:

        if model not in scores_per_technique:
            scores_per_technique[model] = list()
        for size in range(len(test_sizes)):

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_sizes[size])
            regressor = models[model]()
            regressor.fit(x_train, y_train)
            y_pred = regressor.predict(x_test)

            score = regressor.score(x_test, y_test)

            print(models[model])
            print('MSE: ' + str(mean_squared_error(y_test, y_pred)))
            print('R2_Score: ' + str(score))
            print('-----------------------------------------------------')

            scores_per_technique[model].append([test_sizes[size]*100, score])

    bar_pos = -1.5 * config_dictionary['width']
    pos = 0
    for technique in scores_per_technique:
        print('Plotting results for ' + technique)
        results = scores_per_technique[technique]
        print(results)
        results_numpy = np.array(results)
        config_dictionary['label'] = technique
        config_dictionary['bar_position'] = bar_pos
        config_dictionary['color'] = config_dictionary['colors'][pos]
        x_unique, y_unique = npi.group_by(results_numpy[..., 0]).mean(results_numpy[..., 1])
        x_unique, y_std = npi.group_by(results_numpy[..., 0]).std(results_numpy[..., 1])
        plotter.plot_data_using_error_bars(x_unique, y_unique, y_std, config_dictionary)
        # plotting_tool.plot_data_using_bars(x_unique, y_unique, config_dictionary)
        bar_pos += config_dictionary['width']
        pos += 1

    plotter.store_and_show(**config_dictionary)

    # print mean_squared_error(y_test, y_pred)
    # print rfr.score(x_test, y_test)


def norm(df):
    result = df / (max(df))
    return result


if __name__ == '__main__':
    print("ok")
    df = loadData()
    predict(df)