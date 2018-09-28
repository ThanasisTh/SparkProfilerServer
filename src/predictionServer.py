import datetime
import os
import matplotlib
import pandas as pd
import sys
import dateutil.parser
from flask import request, json
from pymongo import MongoClient

from src import app


matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import BaggingRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import normalize

partial_stage_models = {}
partial_stage_models_unique_id = 0

whole_models = {}
whole_models_unique_id = 0

@app.route('/getExecTime', methods=['POST'])
def getExecTime():
    global whole_models_unique_id
    data = request.json

    client = MongoClient('localhost', 27017)
    print(str(client))
    db = client['dioneJson']
    collection = db['Stuff']

    name = data['appName']

    data_headers = {'executors': [], 'time': [], 'inputSplit': [], 'executorMemory': [], 'cores': [], 'stages': []}
    get_exec_times_query = {"application_name": name}
    docs = collection.find(get_exec_times_query)
    exec_sums = []
    parser = dateutil.parser
    for doc in docs:
        attempts = doc['attempts'][0]
        a = parser.parse(attempts['startTime'])
        b = parser.parse(attempts['endTime'])
        exec_sums.append(int((b-a).total_seconds()))

        environment_list = doc['environment']
        execMemory = environment_list['spark@executor@memory']
        split = environment_list['spark@split']
        executors = doc['executors']
        cores = environment_list['spark@cores@max']
        stages = len(doc['stages'])
        time = int((b-a).total_seconds())

        data_headers['inputSplit'].append(float(split.replace('@', '.')))
        data_headers['executorMemory'].append(int(execMemory[:-1]))
        data_headers['executors'].append(len(executors))
        data_headers['time'].append(time)
        data_headers['cores'].append(cores)
        data_headers['stages'].append(stages)

    df = pd.DataFrame.from_dict(data_headers)
    print('got data, preparing to train model')
    y = df['time']
    x = df.drop('time', 1)

    regressor = BaggingRegressor()
    fit = regressor.fit(x, y)

    print('trained, now saving model')
    print('done, returning')
    whole_models[whole_models_unique_id] = regressor
    whole_models_unique_id += 1
    return str(whole_models_unique_id) + ',' + str(sum(exec_sums))

    #      get all attempt time diff for app name
    #      create model for whole app
    #  return model_id, sum(exec_times)




@app.route('/profiler/stages', methods=['POST'])
def predict_stages():
    data = request.json

    id = data['predictionModelId']
    features = data['features']
    regressor = partial_stage_models[id]
    y_pred = regressor.predict([features])

    print(y_pred)

    return str(y_pred[0])

@app.route('/profiler/whole', methods=['POST'])
def predict_whole():
    data = request.json

    id = data['predictionModelId']
    features = data['features']
    regressor = whole_models[id]
    y_pred = regressor.predict([features])

    print(y_pred)

    return str(y_pred[0])


@app.route('/train_model', methods=['POST'])
def train():
    global partial_stage_models_unique_id
    data = request.json

    client = MongoClient('localhost', 27017)
    print(str(client))
    db = client['dioneJson']
    collection = db['Stuff']

    name = data['appName']
    stage = data['stageId']

    data_headers = {'executors': [], 'time': [], 'inputSplit': [], 'executorMemory': [], 'cores': []}
    get_stage_docs_query = [{"$match": {"application_name": name}}, {"$addFields": {"stages": {
        "$filter": {"input": "$stages", "as": "stages", "cond": {"$eq": ["$$stages.stageID", stage]}}}}}]
    docs = collection.aggregate(get_stage_docs_query)

    for doc in docs:
        try:
            stage_list = doc['stages'][0]
            environment_list = doc['environment']

            time = int(stage_list['time'])
            execMemory = environment_list['spark@executor@memory']
            split = environment_list['spark@split']
            executors = doc['executors']
            cores = environment_list['spark@cores@max']

            data_headers['inputSplit'].append(float(split.replace('@', '.')))
            data_headers['executorMemory'].append(int(execMemory[:-1]))
            data_headers['executors'].append(len(executors))
            data_headers['time'].append(time)
            data_headers['cores'].append(cores)
        except IndexError:
            pass

    df = pd.DataFrame.from_dict(data_headers)
    print('got data, preparing to train model')
    y = df['time']
    x = df.drop('time', 1)

    regressor = BaggingRegressor()
    fit = regressor.fit(x, y)

    print('trained, now saving model')

    path = 'partial_stage_models/' + name + '/stage_' + str(stage) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    fl = path + str(partial_stage_models_unique_id) + '.pkl'
    joblib.dump(fit, fl)

    print('done, returning')
    partial_stage_models[partial_stage_models_unique_id] = regressor
    partial_stage_models_unique_id += 1
    return str(partial_stage_models_unique_id)


@app.route('/test')
def test():
    return 'Server works!'


def run():
    app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True)
    # name = 'Als Example'
    # client = MongoClient('localhost', 27017)
    # print(str(client))
    # db = client['dioneJson']
    # collection = db['AllApps']
    # data_by_stages = {}
    # # get count of apps with given name
    # # find_max_stages_query = [{"$match": {"application_name": "AlsDataframe-Application"}},
    # #                           {"$project": {"name": "$application_name",
    # #                                         "numberOfStages": {"$size": "$stages.stageID"}}},
    # #                           {"$sort": {"count": -1}},
    # #                           {"$limit": 1}
    # #                         ]
    #
    # # get all apps with given name
    # find_all_query = {"application_name": name}
    # stage_ids = []
    # result_doc = list(collection.find(find_all_query))
    # score_list = []
    #
    # # get all stage ids from every execution
    # for doc in result_doc:
    #     for stage in doc['stages']:
    #         if stage['stageID'] not in stage_ids:
    #             stage_ids.append(stage['stageID'])
    #
    # # get data from all executions for each stage
    # for stage in stage_ids:
    #
    #     data_headers = {'executors': [], 'time': [], 'inputSplit': [], 'executorMemory': [], 'cores': []}
    #     if stage not in data_by_stages.keys():
    #         data_by_stages[stage] = data_headers
    #     get_stage_docs_query = [{"$match": {"application_name": name}}, {"$addFields": {"stages": {
    #         "$filter": {"input": "$stages", "as": "stages", "cond": {"$eq": ["$$stages.stageID", stage]}}}}}]
    #     docs = collection.aggregate(get_stage_docs_query)
    #
    #     for doc in docs:
    #         try:
    #             stage_list = doc['stages'][0]
    #             environment_list = doc['environment']
    #
    #             time = int(stage_list['time'])
    #             numOfTasks = stage_list['numOfTasks']
    #             execMemory = environment_list['spark@executor@memory']
    #             split = environment_list['spark@split']
    #             executors = doc['executors']
    #             cores = environment_list['spark@cores@max']
    #
    #             data_by_stages[stage]['inputSplit'].append(float(split.replace('@', '.')))
    #             data_by_stages[stage]['executorMemory'].append(int(execMemory[:-1]))
    #             data_by_stages[stage]['executors'].append(len(executors))
    #             # data_by_stages[stage]['numOfTasks'].append(numOfTasks)
    #             data_by_stages[stage]['time'].append(time)
    #             data_by_stages[stage]['cores'].append(cores)
    #         except IndexError:
    #             pass
    #
    #     df = pd.DataFrame.from_dict(data_by_stages[stage])
    #     size = len(df)
    #     file, x_test, y_test = train(df, stage, name)
    #
    #     score_list.append(predict(file, (x_test, y_test)))
    #
    # # max_score = max(score_list)
    # # print max(score_list)
