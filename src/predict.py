import json
import os

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
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import BaggingRegressor
from src.plot.plotter import Plotter
from sklearn.externals import joblib

models = {"Linear Regression": LinearRegression, "Random Forest Regressor": RandomForestRegressor,
          "SGD Regressor": SGDRegressor, "Bayesian Ridge": BayesianRidge,
          "Bagging Regressor": BaggingRegressor, "Kernel Ridge": KernelRidge, "Decision Tree": DecisionTreeRegressor}

dataHeaders = [#'applicationName', 'process',
               'inputBytes', 'inputRecords', 'executors', 'shuffleWrite', 'time']

def loadData():
    client = MongoClient('localhost', 27017)
    print(str(client))
    db = client['dioneJson']
    collection = db['AllApps']

    data_headers = {'inputBytes': [], 'inputRecords': [], 'executors': [], 'numOfTasks': [], 'time': []}
    data_by_stages = {}
    stages_by_app = {}

    for doc in collection.find():
        name = doc['application_name']
        executors = doc['executors']
        stages = doc['stages']
        attempts = doc['attempts']
        if name not in stages_by_app:
            stages_by_app[name] = {}

        for stage in stages:
            stageID = stage['stageID']
            if stageID is not None:
                if stageID not in data_by_stages:
                    data_by_stages[stageID] = data_headers
                time = stage['time']
                inputBytes = stage['inputBytes']
                outputBytes = stage['outputBytes']
                inputRecords = stage['inputRecords']
                outputRecords = stage['outputRecords']
                numOfTasks = stage['numOfTasks']

                data_by_stages[stageID]['inputBytes'].append(inputBytes)
                data_by_stages[stageID]['inputRecords'].append(inputRecords)
                data_by_stages[stageID]['executors'].append(len(executors))
                data_by_stages[stageID]['numOfTasks'].append(numOfTasks)
                data_by_stages[stageID]['time'].append(time)

            # for executor in executors:
            #     execID = executor['id']
            #     inputSize = executor['input']
            #     shuffleWrite = executor['shuffleWrite']

        stages_by_app[name] = data_by_stages

    simple_apps = pd.DataFrame.from_dict(stages_by_app)

    collection = db['ComplexAppsData']
    for doc in collection.find():
        name = doc['application_name']
        executors = doc['executors']
        stages = doc['stages']
        attempts = doc['attempts']
        if name not in stages_by_app:
            stages_by_app[name] = {}

        for stage in stages:
            stageID = stage['stageID']
            if stageID not in data_by_stages:
                data_by_stages[stageID] = data_headers
            time = stage['time']
            inputBytes = stage['inputBytes']
            outputBytes = stage['outputBytes']
            inputRecords = stage['inputRecords']
            outputRecords = stage['outputRecords']
            numOfTasks = stage['numOfTasks']

            data_by_stages[stageID]['inputBytes'].append(inputBytes)
            data_by_stages[stageID]['inputRecords'].append(inputRecords)
            data_by_stages[stageID]['executors'].append(len(executors))
            data_by_stages[stageID]['numOfTasks'].append(numOfTasks)
            data_by_stages[stageID]['time'].append(time)

        # for executor in executors:
        #     execID = executor['id']
        #     inputSize = executor['input']
        #     shuffleWrite = executor['shuffleWrite']

        stages_by_app[name] = data_by_stages

    complex_apps = pd.DataFrame.from_dict(stages_by_app)

    return simple_apps, complex_apps


def predict(app_frame, complex_frame=None, validate=True):
    config_dictionary = json.load(open('/home/thanasis/PycharmProjects/dionePredict/analysis.json'))

    test_sizes = config_dictionary['train_size']
    scores_per_app = {}
    for app in app_frame.keys():

        total_app_time_pred = {}
        total_app_time_real = {}
        if app not in scores_per_app:
            total_app_time_real[app] = {}
            total_app_time_pred[app] = {}
            scores_per_app[app] = {}
        real_per_stage = {}
        for model in models:
            scores_per_technique = {}
            if model not in scores_per_technique:
                scores_per_technique[model] = []
                total_app_time_pred[app][model] = {}
                total_app_time_real[app][model] = {}
                scores_per_app[app][model] = []

            for size in range(len(test_sizes)):
                if size not in total_app_time_real[app][model]:
                    total_app_time_real[app][model][size] = 0
                    total_app_time_pred[app][model][size] = 0
                predictions_per_stage = {}
                for stageID in app_frame[app].keys():
                    stage = pd.DataFrame.from_dict(app_frame[app][stageID])
                    for col in stage:
                        stage[col] = norm(stage[col])

                    if stageID not in predictions_per_stage:
                        predictions_per_stage[stageID] = []
                        real_per_stage[stageID] = []

                    y = stage['time']
                    x = stage.drop('time', 1)

                    if validate:
                        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_sizes[size])

                    try:
                        regressor = models[model]()
                        regressor.fit(x_train, y_train)

                        # path = 'models/' + app + '/stage_' + str(stageID) + '/' + model + '/' + str(test_sizes[size])
                        # if not os.path.exists(path):
                        #     os.makedirs(path)
                        # fl = path + '/model.pkl'
                        # print(regressor)
                        # print(fl)

                        # joblib.dump(fit, fl)

                        y_pred = regressor.predict(x_test)

                        predictions_per_stage[stageID] = y_pred
                        real_per_stage[stageID] = y_test.values
                        total_app_time_pred[app][model][size] += predictions_per_stage[stageID]
                        total_app_time_real[app][model][size] += real_per_stage[stageID]
                    except MemoryError:
                        print("Memory error for model: " + model)
                        del regressor
                        pass

                scores_per_app[app][model].append([test_sizes[size]*100, r2_score(total_app_time_real[app][model][size], total_app_time_pred[app][model][size])])

    # TODO: Fix to include other sizes, store results for everything, make multiple figures
    for app in app_frame.keys():
        plotter = Plotter()
        plotter.setup_plot(**config_dictionary)
        pos = 0
        bar_pos = -1.5 * config_dictionary['width']
        for technique in scores_per_app[app].keys():
            print('Plotting results for ' + technique)
            results = scores_per_app[app][technique]
            print(results)
            results_numpy = np.array(results)
            config_dictionary['label'] = technique
            config_dictionary['bar_position'] = bar_pos
            config_dictionary['color'] = config_dictionary['colors'][pos]
            x_unique, y_unique = npi.group_by(results_numpy[..., 0]).mean(results_numpy[..., 1])
            x_unique, y_std = npi.group_by(results_numpy[..., 0]).std(results_numpy[..., 1])
            plotter.plot_data_using_error_bars(x_unique, y_unique, y_std, config_dictionary)
            bar_pos += config_dictionary['width']
            pos += 1
        config_dictionary['output_file'] = app
        plotter.store_and_show(**config_dictionary)

        # scores_per_app[app].append()


def norm(df):
    df.fillna(value=0)
    result = df / (max(df))
    return result


if __name__ == '__main__':
    df, _ = loadData()
    predict(df)
