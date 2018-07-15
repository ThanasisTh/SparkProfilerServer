import random
from time import time

from sklearn import linear_model, gaussian_process
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import numpy_indexed as npi
import sys
from sklearn.kernel_ridge import KernelRidge
import json
from sklearn import preprocessing
from sklearn import tree
from  math import sqrt

from src.main.python.plots import plotter


class estimation_analysis():

    def evaluateModel(self,technique, train, test):
        if technique == 'Linear Regression':
            # Create linear regression object
            regr = linear_model.LinearRegression()
        elif technique == 'Kernel Ridge':
            regr = KernelRidge(alpha=1.0)
        elif technique == 'Ridge Regression':
            regr = linear_model.Ridge(alpha = .7)
        elif technique == 'Decision Tree':
            regr = tree.DecisionTreeRegressor(max_depth=10, min_samples_leaf=1)
        elif technique == "Random Forest":
            regr = RandomForestRegressor()
        elif technique == 'Gaussian Process':
            kernel= RationalQuadratic(length_scale=1.0, alpha=100)

            regr = gaussian_process.GaussianProcessRegressor(kernel=kernel,alpha=1)

        # min_max_scaler = preprocessing.MaxAbsScaler()
        #
        # train = min_max_scaler.fit_transform(train)
        # test = min_max_scaler.fit_transform(test)

        # train = preprocessing.scale(train)
        # test = preprocessing.scale(test)

        print('Total dataset size: ', len(self.data_array))
        print('Train points', len(train))
        print('Test points', len(test))


        train_data_X = train[:, :-1]
        train_data_X[:,0]=train_data_X[:,0]/np.max(train_data_X[:,0])
        train_data_X[:,1]=train_data_X[:,1]/np.max(train_data_X[:,1])
        train_data_Y = train[:, -1]

        test_data_X = test[:, :-1]
        test_data_X[:,0]=test_data_X[:,0]/np.max(test_data_X[:,0])
        test_data_X[:,1]=test_data_X[:,1]/np.max(test_data_X[:,1])

        test_data_Y = test[:, -1]

        # Train the model using the training sets
        regr.fit(train_data_X, train_data_Y)

        # Make predictions using the testing set
        test_data_Y_predictions = regr.predict(test_data_X)
        print("________________--------------------_________________------------------------")
        print(test_data_Y_predictions)

        print('--------Started-----------')
        for actual, pred in zip(test_data_Y, test_data_Y_predictions):
            print('Actual ' + str(actual) + ', predict ' + str(pred))
        print('--------Ended-----------')

        mean_sq_error = mean_squared_error(test_data_Y, test_data_Y_predictions)
        r2_score_value =r2_score(test_data_Y, test_data_Y_predictions)

        # # The mean squared error
        print("Mean squared error: %.2f" % mean_sq_error)
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score_value)

        test_concat = np.concatenate([test_data_Y, test_data_Y_predictions])

        normalized_mean_squared_error = sqrt(mean_sq_error) / (sum(test_concat) / len(test_concat))
        print(normalized_mean_squared_error)
        return [self.split * 100, r2_score_value]

    def main(self):
        if len(sys.argv) != 2:
            print('Please provide at least one argument.')
            exit(-1)

        file_handle = open(sys.argv[1])
        config_dictionary = json.load(file_handle)


        if 'inputFile' not in config_dictionary or 'train_size' not in config_dictionary:
            print('Please specify the input files and the percentage of test data to consider!')
            exit(-1)
        name_pos=0
        # output_file_handler = open(config_dictionary['results_file'], 'w')
        for input_file in config_dictionary['inputTimeFile']:
            config_dictionary["output_file"]= config_dictionary["output_files"][name_pos]
            name_pos+=1
            plotting_tool = plotter.Plotter()
            plotting_tool.setup_plot(**config_dictionary)
            self.data_array = np.genfromtxt(input_file, delimiter=',', skip_header=False)
            self.data_array = createSynthMatrixRowsExtender(self.data_array,1,10)
            split_sizes = config_dictionary['train_size']
            iterations = config_dictionary['iterations']
            per_technique_dict = dict()
            per_technique_times_dict = dict()
            for self.split in split_sizes:
                for i in range(0, iterations):
                    train, test = train_test_split(self.data_array, train_size=self.split, shuffle=True)
                    for technique in config_dictionary['techniques']:
                        if technique not in per_technique_dict:
                            per_technique_dict[technique] = list()
                            per_technique_times_dict[technique]=list()
                        print('Considering', technique)
                        t0=time()
                        per_technique_dict[technique].append(self.evaluateModel(technique, train, test))
                        t1=time()
                        per_technique_times_dict[technique].append(t0-t1)
            bar_pos = -1.5 * config_dictionary['width']
            pos = 0
            for technique in per_technique_dict:
                print('Plotting results for ' + technique)
                results = per_technique_dict[technique]
                results_numpy = np.array(results)
                config_dictionary['label'] = technique
                config_dictionary['bar_position'] = bar_pos
                config_dictionary['color'] = config_dictionary['colors'][pos]
                x_unique, y_unique = npi.group_by(results_numpy[..., 0]).mean(results_numpy[..., 1])
                x_unique, y_std = npi.group_by(results_numpy[..., 0]).std(results_numpy[..., 1])
                plotting_tool.plot_data_using_error_bars(x_unique, y_unique, y_std, config_dictionary)
                # plotting_tool.plot_data_using_bars(x_unique, y_unique, config_dictionary)
                bar_pos += config_dictionary['width']
                pos += 1

            plotting_tool.store_and_show(**config_dictionary)


def createSynthMatrixRowsExtender(input_matrix,column=1,size=1):

    x=input_matrix





    # for i in range(0,int(row_repetition)):
    #     x=np.concatenate([x,x_1],axis=1)


    rows=x.shape[0]
    for i in range(0,size):

        for r in range(0,rows):

            n= random.uniform(0.9,1.1)
            temp = x[r,:]
            temp=np.reshape(temp,(temp.shape[0],1)).transpose()

            temp[0,column]= temp[0,column]*n

            x = np.concatenate([x, temp], axis=0)


    return x


if __name__ == '__main__':
    estimation_analysis().main()
