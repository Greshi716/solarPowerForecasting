import pandas as pd
import numpy as np
from feature_engine.selection import DropConstantFeatures,DropDuplicateFeatures
from sklearn.pipeline import Pipeline
from feature_engine.selection import SmartCorrelatedSelection
from application_logging.logger import App_Logger

class Preprocessor:
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object=logger_object
    
    def check_for_missingvalues(self,data):
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        try:
            self.null_counts=data.isna().sum()
            for i in self.null_counts:
                if i>0:
                    self.null_present=True
                    break
            if(self.null_present):
                dataframe_with_null=pd.DataFrame()
                dataframe_with_null['columns']=data.columns
                dataframe_with_null['missing values count']=np.asarray(data.isna().sum())
                dataframe_with_null.tocsv('preprocessing_data/null_values.csv')
            self.logger_object.log(self.file_object,
                               'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present,np.asarray(data.columns)
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception() 
    
    def filling_missing_values(self,data,columns):
        self.logger_object.log(self.file_object,'Entered filling_missing_values method of Preprocessor class')
        self.data=data
        try:
            for column in columns:
                data[columns[column]].fillna(data[columns[column]].mean(),inplace=True)
            self.logger_object.log(self.file_object,
                                   'Missing Values replaced with mean successfully. Exited the impute_missing_values method of the Preprocessor class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()
            
    def seperate_features_target(self,data,label_column_name):
        self.logger_object.log(self.file_object,'Entered the seperate_label_feature method of the Preprocessor class')
        try:
            print("data is : ")
            print(data)
            self.X=data.drop(labels=label_column_name,axis=1)
            self.Y=data[label_column_name]
            self.logger_object.log(self.file_object,
                                    'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,
                                    'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(
                                        e))
            self.logger_object.log(self.file_object,
                                    'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()
    
    def get_maxoccurance(self,model):
        self.logger_object.log(self.file_object,'Entered the columns_to_drop method of the Preprocessor class')
        # print("model of column")
        # print(model)
        try:
            # print("model of column try:")
            # print(model)
            maxocczeinth=model['zenith'].value_counts().idxmax()
            maxoccazimuth=model['azimuth'].value_counts().idxmax()

            # print("modeld:")
            # print(maxoccazimuth)
            self.logger_object.log(self.file_object,
                                    'Maxoccurance found!. Exited the get_maxoccurance method of the Preprocessor class')
            return maxoccazimuth,maxocczeinth
        except Exception as e:
            self.logger_object.log(self.file_object,
                                    'Exception occured in columns_to_drop method of the Preprocessor class. Exception message:  ' + str(
                                        e))
            self.logger_object.log(self.file_object,
                                    'Dropping unsuccessful. Exited the columns_to_drop method of the Preprocessor class')
            raise Exception()

    def features_selection(self,x): 
        pip=Pipeline([
                    ('Constant',DropConstantFeatures(tol=0.99)),
                    ('Duplicate',DropDuplicateFeatures()),       
        ])
        x_new=pip.fit_transform(x)
        sel=SmartCorrelatedSelection(selection_method='variance')
        x_new1=sel.fit_transform(x_new)
        return x_new1

