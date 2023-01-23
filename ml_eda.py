import pandas as pd
import numpy as np
import math
import sklearn.datasets
import ipywidgets as widgets

##Seaborn for fancy plots. 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8,8)

class edaDF2:
    # Attributes
    #----------
    #data : dataframe a dataframe on which the EDA will be performed
    #data
     def __init__(self, dataframe):
        self.df = dataframe

     def info(self):
        return self.df.info()
     
     def describe(self):
        return self.df.describe()
     
     def plot_histograms(self):
        self.df['target'].hist(bins=50, figsize=(20,15))
        plt.show()

     def plot_correlation_matrix(self):
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True)
        plt.show()
     
     def plot_countplot(self, column):
        plt.figure(figsize=(12,6))
        sns.countplot(x=column, data=self.df)
        plt.show()
     
        #generates countplots for the categorical variables in the dataset 
     def plot_boxplot(self, column):
        plt.figure(figsize=(12,6))
        sns.boxplot(x=column, data=self.df)
        plt.show()
    
     def plot_barplot(self, x_col, y_col):
        plt.figure(figsize=(12,6))
        sns.barplot(x=x_col, y=y_col, data=self.df)
        plt.show()
    
     def fullEDA(self):
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out = widgets.Output()
        

        tab = widgets.Tab(children = [out1, out2, out3])
        tab.set_title(0, 'Info')
        tab.set_title(1, 'Categorical')
        tab.set_title(2, 'Numerical')
        display(tab)

        with out1:
            self.info()

        with out2:
            fig2 = self.countPlots(splitTarg=True, show=False)
            plt.show(fig2)
        
        with out3:
            fig3 = self.histPlots(kde=True, show=False)
            plt.show(fig3)