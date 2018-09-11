from src.predictionServer import *

if __name__ == '__main__':

    name = 'Als Example'
    data_by_stages = {}

    client = MongoClient('localhost', 27017)
    print(str(client))
    db = client['dioneJson']
    collection = db['AllApps']

    predicted_time_totals = []
    actual_time_total = []

    for stage in range(0, 22):
        check_path = 'models/' + name + '/stage_' + str(stage)
        path = 'models/' + name + '/stage_' + str(stage) + '/' + str(stage) + '.pkl'
        if os.path.isdir(check_path):
            data_headers = {'executors': 8, 'time': 60, 'inputSplit': 0.1, 'executorMemory': 11, 'cores': 56}

            df = pd.DataFrame(data_headers, index=[0])
            y = df['time']
            x = df.drop('time', 1)
            _, _, y_pred = predict(path, (x, y))

            predicted_time_totals.append(y_pred)

    print(predicted_time_totals[-1])
    print(sum(predicted_time_totals[:-1]) + predicted_time_totals[-1]*100)
    print(119*y.values*1000)

