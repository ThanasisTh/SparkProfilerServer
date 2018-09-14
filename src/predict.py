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
from sklearn.preprocessing import OneHotEncoder

models = {"Linear Regression": LinearRegression(fit_intercept=False), "Random Forest Regressor": RandomForestRegressor(),
           "Bayesian Ridge": BayesianRidge(),
          "Bagging Regressor": BaggingRegressor(), "Kernel Ridge": KernelRidge(), "Decision Tree": DecisionTreeRegressor()}

dataHeaders = [#'applicationName', 'process',
               'executors', 'time', 'inputSplit', 'executorMemory', 'cores', 'application_id']

def loadData():
    client = MongoClient('localhost', 27017)
    print(str(client))
    db = client['dioneJson']
    collection = db['AllApps']
    name = 'Als Example'

    encoder = OneHotEncoder()
    data_headers = {'application_id': [], 'executors': [], 'time': [], 'inputSplit': [], 'executorMemory': [], 'cores': []}
    data_by_stages = {}
    stages_by_app = {name: {}}

    find_all_query = {"application_name": name}
    stage_ids = []
    result_doc = list(collection.find(find_all_query))

    # get all stage ids from every execution
    for doc in result_doc:
        for stage in doc['stages']:
            if stage['stageID'] not in stage_ids:
                stage_ids.append(stage['stageID'])

    # get data from all executions for each stage
    for stage in stage_ids:

        get_stage_docs_query = [{"$match": {"application_name": name}}, {"$addFields": {"stages": {
            "$filter": {"input": "$stages", "as": "stages", "cond": {"$eq": ["$$stages.stageID", stage]}}}}}]
        docs = collection.aggregate(get_stage_docs_query)
        for doc in docs:
            if stage is not None:
                try:
                    if stage not in data_by_stages:
                        data_by_stages[stage] = {key: [] for key in data_headers.keys()}
                    stage_list = doc['stages'][0]
                    environment_list = doc['environment']

                    id = doc['application_id']
                    time = int(stage_list['time'])
                    execMemory = environment_list['spark@executor@memory']
                    split = environment_list['spark@split']
                    executors = doc['executors']
                    cores = environment_list['spark@cores@max']

                    data_by_stages[stage]['application_id'].append(id)
                    data_by_stages[stage]['inputSplit'].append(float(split.replace('@', '.')))
                    data_by_stages[stage]['executorMemory'].append(int(execMemory[:-1]))
                    data_by_stages[stage]['executors'].append(len(executors))
                    data_by_stages[stage]['time'].append(time)
                    data_by_stages[stage]['cores'].append(cores)
                except IndexError:
                    pass

            # for executor in executors:
            #     execID = executor['id']
            #     inputSize = executor['input']
            #     shuffleWrite = executor['shuffleWrite']

        stages_by_app[name] = data_by_stages

    simple_apps = pd.DataFrame.from_dict(stages_by_app)

    return simple_apps


def predict(app_frame):
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
                    total_app_time_real[app][model][size] = {}
                    total_app_time_pred[app][model][size] = {}
                predictions_per_stage = {}
                for stageID in app_frame[app].keys():
                    stage = pd.DataFrame.from_dict(app_frame[app][stageID])
                    # for col in stage:
                    #     stage[col] = norm(stage[col])

                    if stageID not in predictions_per_stage:
                        predictions_per_stage[stageID] = {}
                        real_per_stage[stageID] = {}

                    ids = stage['application_id']
                    y = stage['time']
                    x = stage.drop(['time', 'application_id'], 1)

                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_sizes[size])

                    regressor = models[model]
                    regressor.fit(x_train, y_train)

                    # path = 'models/' + app + '/stage_' + str(stageID) + '/' + model + '/' + str(test_sizes[size])
                    # if not os.path.exists(path):
                    #     os.makedirs(path)
                    # fl = path + '/model.pkl'
                    # print(regressor)
                    # print(fl)

                    # joblib.dump(fit, fl)

                    y_pred = regressor.predict(x_test)

                    for id in range(len(ids)):
                        if id not in total_app_time_pred[app][model][size].keys():
                            total_app_time_pred[app][model][size][ids[id]] = 0
                            total_app_time_real[app][model][size][ids[id]] = 0
                        if id not in range(len(y_test.values)):
                            real_per_stage[stageID][ids[id]] = 0
                            predictions_per_stage[stageID][ids[id]] = 0
                        else:
                            real_per_stage[stageID][ids[id]] = y_test.values[id]
                            predictions_per_stage[stageID][ids[id]] = y_pred[id]

                        total_app_time_pred[app][model][size][ids[id]] += predictions_per_stage[stageID][ids[id]]
                        total_app_time_real[app][model][size][ids[id]] += real_per_stage[stageID][ids[id]]

                real_list = [total_app_time_real[app][model][size][id] for id in total_app_time_real[app][model][size].keys()]
                pred_list = [total_app_time_pred[app][model][size][id] for id in total_app_time_pred[app][model][size].keys()]

                scores_per_app[app][model].append([test_sizes[size]*100, r2_score(real_list, pred_list)])

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
    df = loadData()
    predict(df)
