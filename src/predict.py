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
from sklearn.metrics import r2_score
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

dataHeaders = [#'applicationName', 'process',
               'inputBytes', 'inputRecords', 'executors', 'shuffleWrite', 'time']

def loadData():
    client = MongoClient('localhost', 27017)
    print(str(client))
    db = client['dioneJson']
    collection = db['SparkExecutors']

    data_headers = {'inputBytes': [], 'inputRecords': [], 'executors': [], 'numOfTasks': [], 'time': []}
    data_by_stages = {}
    stages_by_app = {}
    df = pd.DataFrame(columns=dataHeaders)

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

    app_frame = pd.DataFrame.from_dict(stages_by_app)

    return app_frame


def predict(app_frame):
    for app in app_frame.keys():
        scores_per_app = {}
        total_app_time_pred = {}
        total_app_time_real = {}
        if app not in scores_per_app:
            total_app_time_real[app] = {}
            total_app_time_pred[app] = {}
            scores_per_app[app] = []
        real_per_stage = {}
        for model in models:

            scores_per_technique = {}
            if model not in scores_per_technique:
                scores_per_technique[model] = list()
                total_app_time_pred[app][model] = float()
                total_app_time_real[app][model] = float()

            config_dictionary = json.load(open('/home/thanasis/PycharmProjects/dionePredict/analysis.json'))
            plotter = Plotter()
            test_sizes = config_dictionary['train_size']
            plotter.setup_plot(**config_dictionary)
            for size in range(len(test_sizes)):
                for stageID in app_frame[app].keys():

                    stage = pd.DataFrame.from_dict(app_frame[app][stageID])
                    for col in stage:
                        stage[col] = norm(stage[col])

                    predictions_per_stage = {}
                    if stageID not in predictions_per_stage:
                        predictions_per_stage[stageID] = []
                        real_per_stage[stageID] = []

                    y = stage['time']
                    x = stage.drop('time', 1)

                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_sizes[size])
                    regressor = models[model]()
                    regressor.fit(x_train, y_train)
                    y_pred = regressor.predict(x_test)

                    score = regressor.score(x_test, y_test)

                    predictions_per_stage[stageID] = y_pred
                    real_per_stage[stageID] = y_test
                    # print(models[model])
                    # print('MSE: ' + str(mean_squared_error(y_test, y_pred)))
                    # print('R2_Score: ' + str(score))
                    # print('-----------------------------------------------------')

                    scores_per_technique[model].append([test_sizes[size]*100, score])

                    # predict scores per stage
                    total_app_time_pred[app][model] += predictions_per_stage[stageID]
                    total_app_time_real[app][model] += real_per_stage[stageID]

                scores_per_app[app][model] = [test_sizes[size]*100, r2_score(total_app_time_real[app][model], total_app_time_pred[app][model])]

            bar_pos = -1.5 * config_dictionary['width']
            pos = 0
            for technique in scores_per_technique:
                print('Plotting results for ' + technique)
                results = scores_per_app[technique]
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

            plotter.store_and_show(**config_dictionary)

        # scores_per_app[app].append()


def norm(df):
    result = df / (max(df))
    return result


if __name__ == '__main__':
    print("ok")
    df = loadData()
    predict(df)