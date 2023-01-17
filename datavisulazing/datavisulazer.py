import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from application_logging.logger import App_Logger

class datavisulazer: 
    def shape(self,data):
        return data.shape
    
    def describe(self,data):
        return data.describe()
    
    def histogram(self,data):
        data.hist(figsize=(20,20))
        plt.savefig('graphs/histogram.PNG')

    def boxplot(self,data):
        lst=pd.Series.tolist (data.columns)
        plt.figure(figsize=(50,50))
        data.boxplot(lst)
        plt.savefig('graphs/boxplot.PNG')
    
    def heatmap(self,data):
        plt.figure(figsize=(20,20))
        sns.heatmap(data.corr(),annot=True)
        plt.savefig('graphs/heatmap.PNG')

