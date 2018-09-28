import matplotlib
import numpy as np
from matplotlib import pyplot as plt

pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
plt.rcParams.update(pgf_with_rc_fonts)
import os

class Plotter:
    """
    The class that will be used for plotting figures using
    the matplotlib library.
    """

    def __init__(self):
        """
        The default constructor.
        """
        self.figure = None
        self.ax = None

    def setup_plot(self, **kwargs):
        """
        The function that will setup all the plotting details (e.g., fontsize, labels, etc)
        :param kwargs: the dictionary that will keep all the arguments.
        :return: None.
        """
        if 'general_font_size' in kwargs:
            font = {'family': 'sans-serif', 'serif': ['Helvetica'], 'size': kwargs['general_font_size']}
            matplotlib.rc('font', **font)
        matplotlib.rcParams['text.usetex'] = True

        figure = plt.figure()
        if 'figure_width' in kwargs and 'figure_height' in kwargs:
            figure.set_size_inches(kwargs['figure_width'], kwargs['figure_height'])

        if 'title' in kwargs:
            figure.suptitle(kwargs['title'], fontsize=12)
        ax = figure.add_subplot(111)

        if 'x_tic_size' in kwargs:
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(kwargs['x_tic_size'])



        if 'y_tic_size' in kwargs:
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(kwargs['x_tic_size'])

        if 'x_lim' in kwargs:
            ax.set_xlim(kwargs['x_lim'])
            start, end = ax.get_xlim()
            if 'x_step' in kwargs:
                ax.xaxis.set_ticks(np.arange(start, end, kwargs['x_step']))

        if 'x_ticks' in kwargs:
            ax.set_xticks(kwargs['x_ticks'])

        if 'y_lim' in kwargs:
            print('Adding y limit')
            ax.set_ylim(kwargs['y_lim'])
            start, end = ax.get_ylim()
            print(start, end)
            if 'y_step' in kwargs:
                ax.yaxis.set_ticks(np.arange(start, end, kwargs['y_step']))

        if 'use_grid' in kwargs:
            ax.grid(kwargs['use_grid'], linestyle="--", color='black', linewidth=0.2)

        if 'title_name' in kwargs:
            ax.set_title(kwargs['title_name'])

        if 'x_label' in kwargs:
            ax.set_xlabel(kwargs['x_label'], labelpad=1, fontsize=kwargs['label_size'])

        if 'y_label' in kwargs:
            ax.set_ylabel(r'$R^2$ Score', labelpad=1, fontsize=kwargs['label_size'])

        self.figure = figure
        self.ax = ax

    def store_and_show(self, **kwargs):
        """
        The function that will be used for storing the generated figure 
        and also displaying it.
        :param fig: the figure to store.
        :param kwargs: the parameters that can specify the location where the figure
        will be stored.
        :return: None
        """
        plt.tight_layout()


        if 'legend_font_size' in kwargs:
            plt.legend(fontsize=kwargs['legend_font_size'], loc="lower right", fancybox=True, framealpha=0.5)
        else:
            plt.legend()

        if 'output_file' in kwargs:
            print('Saving figure in ' +kwargs['output_folder']+kwargs['output_file'])
            if not os.path.exists(kwargs['output_folder']):
                os.makedirs(kwargs['output_folder'])
            self.figure.savefig(kwargs['output_folder']+kwargs['output_file'], format='pdf', dpi=1200)

        if 'should_show_figure' in kwargs:
            if kwargs['should_show_figure']:
                plt.show()
        else:
            # by default we display the image
            plt.show()

    def plot_data_using_bars(self, x_data, y_data, info_dictionary):
        print(info_dictionary['bar_position'])
        self.ax.bar(x_data + info_dictionary['bar_position'],
                    y_data, width=info_dictionary['width'],
                    color=info_dictionary['color'],
                    label=info_dictionary['label'], lw=1.0, edgecolor='black')



    def plot_data_using_bars(self, x_data, y_data):
        # print(info_dictionary['bar_position'])
        self.ax.plot(x_data , y_data ,color="red",lw=1.0)

    def plot_data_using_error_bars(self, x_data, y_data, y_error, info_dictionary):
        self.ax.bar(x_data + info_dictionary['bar_position'],
                    y_data, yerr=y_error, width=info_dictionary['width'],
                    color=info_dictionary['color'],
                    label=info_dictionary['label'], lw=1.0, error_kw=dict(ecolor='black'),
                    edgecolor='black')

    def load_data_and_bar_plot_from_dictionary(self, info_dictionary):
        """
        The function that will be used for plotting the data specified in the given
        dictionary.
        :param ax: the subplot to use.
        :param info_dictionary: the dictionary that contains all the information that need
        to be used for plotting the data.
        :return: None
        """

        data = np.genfromtxt(info_dictionary['file'], delimiter=',', skip_header=1)
        print('Data:', data)
        self.ax.bar(data[..., info_dictionary['x_column']] + info_dictionary['bar_position'],
                    data[..., info_dictionary['y_column']], width=info_dictionary['width'],
                    color=info_dictionary['color'], label=info_dictionary['label'])
