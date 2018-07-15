import numpy as np
import matplotlib.pyplot as plt


class EnergyPlotter:


    def main(self):
        type="wordcount"
        cores="8"

        label1 = "3.4Ghz("+cores+ " Cores)"

        label2 = "2.6Ghz("+cores+ " Cores)"

        label3 = "1.6Ghz("+cores+ " Cores)"

        f, ax = plt.subplots(1)
        if(type=="wordcount"):
            data3_4= np.genfromtxt("/home/thanasis/PycharmProjects/express/power_data_spark_applications.WordcountDataframe_3400.csv",delimiter=",")
            data2_6 = np.genfromtxt("/home/thanasis/PycharmProjects/express/power_data_spark_applications.WordcountDataframe_2600.csv",
                                    delimiter=",")
            data1_6 = np.genfromtxt("/home/thanasis/PycharmProjects/express/power_data_spark_applications.WordcountDataframe_1600.csv",
                                    delimiter=",")
            plt.title("WordCount Dataframe Power Consumption")

        else:
            data3_4 = np.genfromtxt("/home/thanasis/PycharmProjects/express/power_data_AlsDataframe-Application_3400.csv", delimiter=",")
            data2_6 = np.genfromtxt("/home/thanasis/PycharmProjects/express/power_data_AlsDataframe-Application_2600.csv", delimiter=",")
            data1_6 = np.genfromtxt("/home/thanasis/PycharmProjects/express/power_data_AlsDataframe-Application_1600.csv", delimiter=",")
            plt.title("Als Dataframe Power Consumption")

        print(data3_4.shape)
        input3_y=data3_4[:,1]
        input3_x=data3_4[:,0]
        print(data2_6.shape)
        input2_y=data2_6[:,1]
        input2_x=data2_6[:,0]
        print(data1_6.shape)
        input1_y=data1_6[:,1]
        input1_x=data1_6[:,0]
        plt.xlabel("Time (Secs)", fontsize=15)
        plt.ylabel("Energy (W)", fontsize=15)
        # plt.scatter(input_x, input_y,  color='black',s=1)
        ax.plot(input3_x, input3_y, color='red', linewidth=1, label=label1)
        ax.plot(input2_x, input2_y, color='blue', linewidth=1, label=label2)
        ax.plot(input1_x, input1_y, color='green', linewidth=1, label=label3)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.legend(fontsize=15, )
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0)

        f.tight_layout()
        plt.show(f)

if __name__=="__main__":

    EnergyPlotter().main()