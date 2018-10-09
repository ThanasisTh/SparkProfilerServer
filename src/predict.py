import json
import math
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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import BaggingRegressor
from src.plot.plotter import Plotter
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest

models = {"Linear Regression": LinearRegression(fit_intercept=False), "Random Forest Regressor": RandomForestRegressor(),
           "Bayesian Ridge": BayesianRidge(),
          "Bagging Regressor": BaggingRegressor(base_estimator=RandomForestRegressor()), "Kernel Ridge": KernelRidge(), "Decision Tree": DecisionTreeRegressor()}

dataHeaders = [#'applicationName', 'process',
               'executors', 'time', 'inputSplit', 'executorMemory', 'cores', 'application_id']


def loadData(name, data_headers, string_dict):
    client = MongoClient('localhost', 27017)
    print(str(client))
    db = client['dioneJson']
    collection = db['Stuff']

    data_by_stages = {}
    stages_by_app = {name: {}}
    stage_list = []
    find_all_query = {"application_name": name}
    result_doc = list(collection.find(find_all_query))

    # get all stage ids from every execution
    for doc in result_doc:
        for stage in doc['stages']:
            if stage['stageID'] not in stage_list:# and stage['stageID'] in string_dict.keys():
                stage_list.append(stage['stageID'])

    print(len(sorted(stage_ids)))
    print(len(string_dict))
    print('---------------------------------')
    # get data from all executions for each stage
    for stage in stage_list:

        # string = string_dict[stage]

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
                    # data_by_stages[stage]['string'].append(string)
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
    test_sizes = config_dictionary['train_size']
    scores_per_app = {}
    for app in app_frame.keys():
        df = pd.DataFrame([])
        for stage in app_frame[app]:
            df = df.append(pd.DataFrame.from_dict(stage))

        # ids = df['application_id']
        # y_all = df['time']
        # x_all = df.drop(['time', 'application_id'], 1)
        # x_all['cores'] = pd.to_numeric(x_all['cores'])

        # This is probably only useful for app-level model regression, pointless at stage level
        # mlb = MultiLabelBinarizer()
        # lb_encoder = LabelEncoder()
        # operations = []
        # string_list = x_all['string'].tolist()
        # for string in string_list:
        #     for char in string:
        #         if char not in operations:
        #             operations.append(char)
        # lb_encoder.fit(operations)
        # results = []
        # for string in string_list:
        #     results.append(lb_encoder.transform(list(string)))
        # results = np.reshape(results, (-1, len(results)))
        # result = results[0]
        # mlb.fit(result)
        # result = mlb.transform(result)
        #
        # x_all = x_all.drop(['string'], axis=1)
        #
        # encoder = OneHotEncoder(sparse=False)
        # encoder.fit(x_all)
        # x_all = encoder.transform(x_all)
        #
        # x_all = np.concatenate((x_all, result), axis=1)
        #
        # mi = mutual_info_regression(x_all, y_all, random_state=0)
        # mi /= np.max(mi)
        # print(mi)
        # x_all = SelectKBest(scorer, k=2).fit_transform(x_all, y_all)

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

                    if stageID not in predictions_per_stage:
                        predictions_per_stage[stageID] = {}
                        real_per_stage[stageID] = {}

                    ids = stage['application_id']
                    y = stage['time']
                    x = stage.drop(['time', 'application_id'], 1)
                    x['cores'] = pd.to_numeric(x['cores'])

                    # x = x.drop('string', axis=1)

                    # for key in x.keys():
                    #    x[key] = norm(x[key])

                    # lb_encoder = LabelEncoder()
                    # lb_encoder.fit(x['string'])
                    # x['string'] = lb_encoder.transform(x['string'])
                    encoder = OneHotEncoder(sparse=False)
                    encoder.fit(x)
                    x = encoder.transform(x)

                    # half = math.floor(x.shape[1] / 2)
                    #
                    # x = SelectKBest(scorer, k=half).fit_transform(x, y)

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

    for app in app_frame.keys():
        result_matrix = {}
        for technique in scores_per_app[app].keys():
            results = scores_per_app[app][technique]
            result_matrix[technique] = results

        # scores_per_app[app].append()
    return result_matrix


def norm(df):
    df.fillna(value=0)
    result = df / (max(df))
    return result


def scorer(x, y):
    return mutual_info_regression(x, y, random_state=0)


if __name__ == '__main__':
    features = {'application_id': [], 'time': [],
                'executors': [], 'executorMemory': [],
                'cores': [], 'inputSplit': []} #, 'string': []}

    config_dictionary = json.load(open('/home/thanasis/PycharmProjects/dionePredict/analysis.json'))

    names = ['Als Explicit', 'Als Implicit', 'Kmeans App', 'LogisticRegressionSGD']
    for name in names:
        strings = pd.read_csv(name.lower().replace(' ', '_') + '_stages_string_representations.csv')
        stage_ids = strings['#stage_id']
        string_reps = strings['string_representation']
        string_dict = {stage: string for stage, string in zip(stage_ids, string_reps)}

        result_matrix = {name: {x: [[20, 0], [50, 0], [80, 0]] for x in models.keys()}}

        num_samples = 1
        df = loadData(name, features, string_dict)
        for i in range(num_samples):
            scores = predict(df)
            for technique in models.keys():
                i = 0
                for size in result_matrix[name][technique]:
                    size[1] += scores[technique][i][1]
                    i += 1
        for technique in result_matrix[name].keys():
            for size in result_matrix[name][technique]:
                size[1] /= num_samples

        for app in result_matrix.keys():
            config_dictionary['title'] = app
            plotter = Plotter()
            plotter.setup_plot(**config_dictionary)
            pos = 0
            bar_pos = -1.5 * config_dictionary['width']
            for technique in result_matrix[app].keys():
                print('Plotting results for ' + technique)
                results = result_matrix[app][technique]
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
