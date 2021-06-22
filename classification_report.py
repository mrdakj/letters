import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

class ClassificationReport:
    def __init__(self, class_names, report_dir):
        sns.set(font_scale=1.4)
        self.class_names = class_names
        self.report_dir = report_dir

    def confusion_matrix(self, true_labels, predicted_labels, title):
        # save confusion matrix
        confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels, labels=self.class_names)
        # print(confusion_matrix)
        if len(self.class_names) == 2:
            plt.figure(figsize=(9,6))
        else:
            plt.figure(figsize=(18,13))

        ax = plt.subplot()
        sns.heatmap(confusion_matrix, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Prediktovane vrednosti', fontsize=20)
        ax.set_ylabel('Stvarne vrednosti', fontsize=20)
        ax.set_title('Matrica konfuzije', fontsize=20)
        ax.xaxis.set_ticklabels(self.class_names, fontsize=20)
        ax.yaxis.set_ticklabels(self.class_names, rotation='horizontal', fontsize=20)
        plt.savefig(f'{self.report_dir}/{title}.pdf', format='pdf', bbox_inches='tight')
        plt.clf()


    # functions for plotting classification report
    def __show_values(self, pc, fmt="%.2f", **kw):
        '''
        Heatmap with text in each cell with matplotlib's pyplot
        Source: https://stackoverflow.com/a/25074150/395857 
        By HYRY
        '''
        pc.update_scalarmappable()
        ax = pc.axes
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


    def __cm2inch(self, *tupl):
        '''
        Specify figure size in centimeter in matplotlib
        Source: https://stackoverflow.com/a/22787457/395857
        By gns-ank
        '''
        inch = 2.54
        if type(tupl[0]) == tuple:
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)


    def __heatmap(self, AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
        '''
        Inspired by:
        - https://stackoverflow.com/a/16124677/395857 
        - https://stackoverflow.com/a/25074150/395857
        '''

        # Plot it out
        fig, ax = plt.subplots()    
        #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
        c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

        # set tick labels
        #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
        ax.set_xticklabels(xticklabels, minor=False, fontsize=20)
        ax.set_yticklabels(yticklabels, minor=False, fontsize=20)

        # set title and x/y labels
        plt.title(title, fontsize=20)
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)

        # Remove last blank column
        plt.xlim( (0, AUC.shape[1]) )

        # Turn off all the ticks
        ax = plt.gca()    
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        # Add color bar
        plt.colorbar(c)

        # Add text in each cell 
        self.__show_values(c)

        # Proper orientation (origin at the top left instead of bottom left)
        if correct_orientation:
            ax.invert_yaxis()
            ax.xaxis.tick_top()       

        # resize 
        fig = plt.gcf()
        fig.set_size_inches(self.__cm2inch(figure_width, figure_height))


    def __plot_classification_report(self, classification_report, title='', cmap='RdBu'):
        '''
        Plot scikit-learn classification report.
        Extension based on https://stackoverflow.com/a/31689645/395857 
        '''
        lines = classification_report.split('\n')

        classes = []
        plotMat = []
        support = []
        class_names = []

        for line in reversed(lines[2 : (len(lines) - 4)]):
            t = line.strip().split()
            if len(t) < 2: continue
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]
            support.append(int(t[-1]))
            class_names.append(t[0])
            plotMat.append(v)

        xlabel = ''
        ylabel = ''
        xticklabels = ['Preciznost', 'Odziv', 'F1-mera']
        yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
        figure_width = 25
        figure_height = len(class_names) + 7
        correct_orientation = False
        self.__heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)


    def classification_report(self, true_labels, predicted_labels, title):
        # save classification report
        report = metrics.classification_report(true_labels, predicted_labels, labels=self.class_names)
        print(report)
        self.__plot_classification_report(report)
        plt.savefig(f'{self.report_dir}/{title}.pdf', format='pdf', bbox_inches='tight')
        plt.clf()
