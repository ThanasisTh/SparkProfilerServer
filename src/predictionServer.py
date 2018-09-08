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
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib


def predict(file, inputVector):
    regressor = joblib.load(file)
    y_pred = regressor.predict(inputVector[0])

    score = r2_score(y_test, y_pred)
    print(score)


def train(df, stagedId: int, appName):
    # client = MongoClient('localhost', 27017)
    # print(str(client))
    # db = client['dioneJson']
    # collection = db['AllSparkApps']
    # data_by_stages = {}
    # data_headers = {'inputBytes': [], 'inputRecords': [], 'executors': [], 'numOfTasks': [], 'time': []}
    # query = [{"$match": {}}, {"$addFields": {"stages": {"$filter": {"input": "$stages", "as": "stages", "cond": {"$eq": ["$$stages.stageID", 0]}}}}}]
    #
    # for doc in collection.aggregate(query):
    #     stages = doc['stages']
    #     executors = doc['executors']
    #     for stage in stages:
    #         stageID = stage['stageID']
    #         if stageID is not None:
    #             if stageID not in data_by_stages:
    #                 data_by_stages[stageID] = data_headers
    #             time = stage['time']
    #             inputBytes = stage['inputBytes']
    #             outputBytes = stage['outputBytes']
    #             inputRecords = stage['inputRecords']
    #             outputRecords = stage['outputRecords']
    #             numOfTasks = stage['numOfTasks']
    #
    #             data_by_stages[stageID]['inputBytes'].append(inputBytes)
    #             data_by_stages[stageID]['inputRecords'].append(inputRecords)
    #             data_by_stages[stageID]['executors'].append(len(executors))
    #             data_by_stages[stageID]['numOfTasks'].append(numOfTasks)
    #             data_by_stages[stageID]['time'].append(time)
    #
    # df = pd.DataFrame.from_dict(data_by_stages[stagedId])


    y = df['time']
    x = df.drop('time', 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    regressor = LinearRegression()
    fit = regressor.fit(x_train, y_train)

    path = 'models/' + appName + '/stage_' + str(stagedId) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    fl = path + str(stagedId) + '.pkl'
    joblib.dump(fit, fl)

    return fl, x_test, y_test


if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    print(str(client))
    db = client['dioneJson']
    collection = db['AllSparkApps']
    data_by_stages = {}
    data_headers = {'inputBytes': [], 'inputRecords': [], 'executors': [], 'numOfTasks': [], 'time': []}
    data_header_names = data_headers.keys()
    # get count of apps with given name
    # find_max_stages_query = [{"$match": {"application_name": "AlsDataframe-Application"}},
    #                           {"$project": {"name": "$application_name",
    #                                         "numberOfStages": {"$size": "$stages.stageID"}}},
    #                           {"$sort": {"count": -1}},
    #                           {"$limit": 1}
    #                         ]

    # get all apps with given name
    find_all_query = {"application_name": "AlsDataframe-Application"}
    stage_ids = []
    result_doc = list(collection.find(find_all_query))

    # get all stage ids from every execution
    for doc in result_doc:
        print(doc)
        for stage in doc['stages']:
            if stage['stageID'] not in stage_ids:
                stage_ids.append(stage['stageID'])

    for stage in stage_ids:
        get_stage_docs_query = [{"$match": {"application_name": "AlsDataframe-Application"}}, {"$addFields": {"stages": {
            "$filter": {"input": "$stages", "as": "stages", "cond": {"$eq": ["$$stages.stageID", stage]}}}}}]
        docs = collection.aggregate(get_stage_docs_query)

        for doc in docs:
            executors = doc['executors']
            if stage is not None:
                if stage not in data_by_stages:
                    data_by_stages[stage] = data_headers
                try:
                    stage_list = doc['stages'][0]
                except IndexError:
                    print(doc)
                    pass
                time = stage_list['time']
                inputBytes = stage_list['inputBytes']
                outputBytes = stage_list['outputBytes']
                inputRecords = stage_list['inputRecords']
                outputRecords = stage_list['outputRecords']
                numOfTasks = stage_list['numOfTasks']

                data_by_stages[stage]['inputBytes'].append(inputBytes)
                data_by_stages[stage]['inputRecords'].append(inputRecords)
                data_by_stages[stage]['executors'].append(len(executors))
                data_by_stages[stage]['numOfTasks'].append(numOfTasks)
                data_by_stages[stage]['time'].append(time)

        df = pd.DataFrame.from_dict(data_by_stages[stage])

        file, x_test, y_test = train(df, stage, 'AlsDataframe-Application')

        predict(file, (x_test, y_test))